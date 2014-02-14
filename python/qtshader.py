#!/bin/env python
#-------------------------------------------------------------------------------
# Name:        QGLShaderWidget
# Purpose:     Make it easy to use GLSL shaders in a QGLWidget
#
# Author:      dougm
#
# Created:     14/04/2011
# Licence:     None
#-------------------------------------------------------------------------------

"""
A QGLWidget with the addition of named shaders

Adds the methods:
    defineShader()
    applyShader()
    removeShader()
    setShaderParam()
    shaderParams()

 e.g.
    w = QGLShaderWidget()
    w.defineShader("colorize",
        open("lambert.vert").read(),
        open("colorize.frag").read())
    w.setShaderVar("color", (1, 0.8, 1, 1))
    w.applyShader('colorize')
    ...
    w.removeShader()
"""

#-------------------------------------------------------------------------------

from   path              import path as Path
from   PIL               import Image

import math

from   PyQt4             import QtCore, QtGui, QtOpenGL
from   OpenGL.GL         import *
from   OpenGL.GLU        import *

TEXTURE = (GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3,
           GL_TEXTURE4, GL_TEXTURE5, GL_TEXTURE6, GL_TEXTURE7)

try:
    from cgkit.cgtypes import vec3 as Vector3, vec4 as Vector4
except ImportError:
    print "...no cgkit?!?"
    Vector3 = object
    Vector4 = object

class vec2(Vector3):
    def asTuple(self):
        return (self.x,self.y)

class vec3(Vector3):
    def asTuple(self):
        return (self.x,self.y,self.z)

class vec4(Vector4):
    def asTuple(self):
        return (self.x,self.y,self.z,self.w)

#def vec2(f1, f2)         : return (f1, f2)
#def vec3(f1, f2, f3)     : return (f1, f2, f3)
#def vec4(f1, f2, f3, f4) : return (f1, f2, f3, f4)

def mix(c1,c2,m):
    return (1.0-m)*c1 + m*c2

def randColor(sat):
    from random import uniform
    c1 = uniform(1.0-sat,1.0)
    c2 = uniform(1.0-sat,1.0)
    c3 = mix(c1,c2,uniform(0,1))
    c  = [c1,c2,c3]
    shuffle(c)
    c.append(1.0)
    return vec4(*c)

shaderParamDefaults   = {
    'float'       : 0.0,
    'sampler2D'   : '',
    'vec2'        : vec2(0.0,0.0),
    'vec3'        : vec3(0.0,0.0,0.0),
    'vec4'        : vec4(0.0,0.0,0.0,0.0),
}

#-------------------------------------------------------------------------------

class ShaderParam(object):
    """
    A parameter from a GLSL shader

    Has the following attributes:
        name
        decl
        value
    """
    def __init__(self, decl, name, value=None):
        self.decl  = decl
        self.name  = name
        self.value = value or shaderParamDefaults.get(decl,None)
        self.location = 0 # once compiled, will contain address
        self.pixMap   = None # for sampler2D
    def setValue(self,value):
        if not typeCompatible(self.decl,value):
            #print "XXX",shaderParamDefaults[decl]
            raise Exception("value '%s' not acceptable as '%s'" % (value,self.decl))
        self.value = value
        if self.decl == 'sampler2D':
            self.pixMap = QtGui.QPixmap(self.value)
    def __str__(self):
        return "ShaderParam(%s,%s) = %s" % (self.decl,self.name,self.value)
    __repr__ = __str__

def typeCompatible(decl,value):
    if isinstance(value,tuple) and len(value) == 1:
        value = value[0]
    if decl == 'sampler2D':
        if hasattr(value,'startswith'):
            return True
    if isinstance(value,(tuple,list)) and decl.startswith("vec"):
        if len(value) == int(decl[3]):
            return True
    if not isinstance(value,type(shaderParamDefaults[decl])):
        #print "XXX*** not isinstance(value='%s',type(shaderParamDefaults[decl='%s'])='%s')" % (value,decl,type(shaderParamDefaults[decl]),)
        #raise "XXX*** not isinstance("
        return False
    if decl.startswith("vec"):
        if len(value) != len(shaderParamDefaults[decl]):
            #print "XXX*** len(value)='%s' != len(shaderParamDefaults[decl='%s'])='%s'" % (len(value),decl,len(shaderParamDefaults[decl]))
            #raise "XXX*** len(value)="
            return False
    return True

def guessDecl(value):
    """
    Guess the declaration type name given a shader param value
    eg.
        guessDecl( (0,1,0) ) -> 'vec3'
    """
    if isinstance(value, (list,tuple)):
        if len(value) == 2:
            return 'vec2'
        elif len(value) == 3:
            return 'vec3'
        elif len(value) == 4:
            return 'vec4'
    elif isinstance(value, float):
        return 'float'
    # Single int value or named texture.
    elif isinstance(value, (string,unicode,int)):
        return 'sampler2D'


#-------------------------------------------------------------------------------
# https://sites.google.com/site/dlampetest/python/geometry-shaders-from-python
#These three defines exist in OpenGL.GL, but does not correspond to those used here
GL_GEOMETRY_INPUT_TYPE_EXT   = 0x8DDB
GL_GEOMETRY_OUTPUT_TYPE_EXT  = 0x8DDC
GL_GEOMETRY_VERTICES_OUT_EXT = 0x8DDA

_glProgramParameteri = None
def glProgramParameteri( program, pname, value  ):
    global _glProgramParameteri
    if not _glProgramParameteri:
        import ctypes
        # Open the opengl32.dll
        gldll = ctypes.windll.opengl32
        # define a function pointer prototype of *(GLuint program, GLenum pname, GLint value)
        prototype = ctypes.WINFUNCTYPE( ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_int )
        # Get the win gl func adress
        fptr = gldll.wglGetProcAddress( 'glProgramParameteriEXT' )
        if fptr==0:
            raise Exception( "wglGetProcAddress('glProgramParameteriEXT') returned a zero adress, which will result in a nullpointer error if used.")
        _glProgramParameteri = prototype( fptr )
    _glProgramParameteri( program, pname, value )

GL_PATCH_VERTICES = 36466

_glPatchParameteri = None
def glPatchParameteri( pname, value  ):
    global _glPatchParameteri
    if not _glPatchParameteri:
        import ctypes
        # Open the opengl32.dll
        gldll = ctypes.windll.opengl32
        # define a function pointer prototype of *(GLenum pname, GLint value)
        prototype = ctypes.WINFUNCTYPE( ctypes.c_int, ctypes.c_uint, ctypes.c_int )
        # Get the win gl func adress
        fptr = gldll.wglGetProcAddress( 'glPatchParameteriEXT' )
        if fptr==0:
            raise Exception( "wglGetProcAddress('glPatchParameteriEXT') returned a zero adress, which will result in a nullpointer error if used.")
        _glPatchParameteri = prototype( fptr )
    _glPatchParameteri( pname, value )


#-------------------------------------------------------------------------------

class GLShaderProgram(object):
    """
    Contains a set of QGLShader objects and the QGLShaderProgram that links them.
    Has a dict of variables that can be accessed as items:
        sp = GLShaderProgram(w,"phong",v,f)
        sp['spec'] = (1,1,0)
    """
    def __init__(self, parent, name, vtxSource, fragSource, **vars):
        self.parent     = parent # QGLWidget parent
        self.name       = name
        self.vtxSource  = vtxSource
        self.fragSource = fragSource
        self.variables  = dict()
        self._getVariablesFromSource()
        self.variables.update(vars)
        self._restore   = dict() # to restore vars saved when temp overridden
        self.active     = False # if this shader is currently being applied
        self.vtxShader  = None
        self.fragShader = None
        self.program    = None

    def bind(self):
        if self.program == None:
            self.compile()
        self.program.bind()

    def release(self):
        self.program.release()

    def compile(self):
        
        # Compile vertex shader
        if self.vtxSource:
            self.vtxShader = QtOpenGL.QGLShader( QtOpenGL.QGLShader.Vertex, self.parent)
            success = self.vtxShader.compileSourceCode(self.vtxSource)
            if not success:
                raise Exception("compile failed for vtx shader '%s': %s"
                                 % ( self.name, self.vtxShader.log()))
        
        # Compile frag shader
        if self.fragSource:
            self.fragShader = QtOpenGL.QGLShader( QtOpenGL.QGLShader.Fragment, self.parent)
            success = self.fragShader.compileSourceCode(self.fragSource)
            if not success:
                raise Exception("compile failed for frag shader '%s': %s"
                                 % ( self.name, self.fragShader.log()))
        
        # Link the program
        self.program = QtOpenGL.QGLShaderProgram(self.parent)
        if self.vtxSource:
            self.program.addShader(self.vtxShader)
        if self.fragSource:
            self.program.addShader(self.fragShader)
        success = self.program.link()
        if not success:
            raise Exception("link failed for shader '%s': %s"
                             % ( self.name, self.program.log()))

        # Get the address of all the variables
        for varName, var in self.variables.items():
            var.location = self.program.uniformLocation(varName)
            #print "XXX",varName,var,var.location
            if var.location == -1:
                raise Exception("failed to get location of variable '%s' in shader '%s'"
                                 % (varName,self.name,))

    def _getVariablesFromSource(self):
        for source in (self.vtxSource, self.fragSource):
            if not source: continue
            for line in source.split('\n'):
                line = line.strip().replace('  ',' ')
                if line.startswith("uniform"):
                    # assume ';' at end and 2 spaces
                    _, decl, name = line[:-1].split(" ",2)
                    decl = decl.strip()
                    name = name.strip()
                    self.variables[name] = ShaderParam(decl,name)

    def __getitem__(self, name):
        if name not in self.variables:
            raise Exception("shader '%s' can't get variable '%s'" % (self.name,name))
        return self.variables[name]

    def __setitem__(self, name, value):
        if name not in self.variables:
            raise Exception("shader '%s' can't set variable '%s'" % (self.name,name))
        self.variables[name].setValue(value)
        self.parent.sendShaderVars()

    def saveVariables(self):
        for k in self.variables:
            self._restore[k] = self.variables[k]

    def restoreVariables(self):
        for k in self.variables:
            self.variables[k] = self._restore[k]

    def dump(self):
        for x in self.variables.values():
            print x

    def __del__(self):
        try:
            del(self.vtxShader)
            del(self.fragShader)
            del(self.program)
        except:
            print "...didn't clean up properly?"

#-------------------------------------------------------------------------------

class QGLShaderWidget(QtOpenGL.QGLWidget):

    """
    Adds some functionality to the QGLWidget that makes it able to
    manage GLSL shaders.
    
    usage:
        w = QGLShaderWidget()
        w.defineShader("phong",phongVtx,phongFrag)
        w.applyShader("phong",spec=(1,1,0))
        ...draw objects...
        w.removeShader()
    
    The shader programs are accessable as items:
        w['phong']['spec'] = (1,1,1)
    """

    def __init__(self,parent=None):
        super(QGLShaderWidget,self).__init__(parent)
        self.shaderPrograms = dict()
        self.activeShader = None # name of active shader

    def defineShader(self,name,vtxSource=None,fragSource=None,**variables):
        if vtxSource==None and fragSource==None:
            raise Exception("null source for shader named '%s'" % name)
        self.shaderPrograms[name] = GLShaderProgram(self,name,vtxSource,fragSource,**variables)

    def __contains__(self,shaderName):
        return shaderName in self.shaderPrograms

    def __getitem__(self,name):
        if name not in self.shaderPrograms:
            raise Exception("shader '%s' not found" % name)
        return self.shaderPrograms[name]

    def sendShaderVars(self):
        "Send shader vars to the shader using the stored values"
        if self.activeShader == None: return
        shad = self.shaderPrograms[self.activeShader]
        for v in shad.variables.values():
            if v.decl == 'sampler2D':
                if v.pixMap != None:
                    glActiveTexture(TEXTURE[3])
                    texId = self.bindTexture(v.pixMap) #,QtOpenGL.MipmapBindOption)
                    glBindTexture(GL_TEXTURE_2D , texId)
                    shad.program.setUniformValue(v.location, 3)
            else:
                try:
                    if isinstance(v.value,(list,tuple)):
                        #print "XXX shad.program.setUniformValue tuple",v.name,v.value
                        shad.program.setUniformValue(v.location, *v.value)
                    elif isinstance(v.value,(vec2,vec3,vec4)):
                        #print "XXX shad.program.setUniformValue vec",v.name,v.value.asTuple()
                        shad.program.setUniformValue(v.location, *(v.value.asTuple()))
                    else:
                        #print "XXX shad.program.setUniformValue",v.name,v.value
                        shad.program.setUniformValue(v.location, v.value)
                except TypeError, desc:
                    print v
                    raise Exception(desc)
                    sys.exit(1)

    def applyShader(self,name=None,**kw):
        """
        Applies the named shader and temporarily overrides any desired variables
        If the name isn't given, it uses 'default', or if no default, and only
        one shader exists, uses it. Otherwise error.
        """
        if self.activeShader != None:
            raise Exception("cannot apply shader '%s' while shader '%s' is already applied" % (name,self.activeShader))
        if name == None: 
            if 'default' in self:
                name = 'default'
            else:
                if len(self.shaderPrograms) == 1:
                    name = self.shaderPrograms.keys()[0]
                else:
                    raise Exception("cannot determine which shader to apply from %s"
                                    % (", ".join(self.shaderPrograms.keys())))
        self.activeShader = name
        program = self.shaderPrograms[self.activeShader]
        program.saveVariables()
        program.bind()
        self.sendShaderVars()

    def removeShader(self):
        if self.activeShader == None:
            raise Exception("cannot remove shader, none applied")
        program = self.shaderPrograms[self.activeShader]
        program.restoreVariables()
        program.release()
        self.activeShader = None

    def __del__(self):
        for program in self.shaderPrograms.values():
            del(program)
            

#-------------------------------------------------------------------------------

class GLTextureImage:
    "NOT USED... A PIL image used as a texture in a shader"
    def __init__(self,filePath):
        self.path = Path(filePath)
        if not self.path.exists():
            raise Exception("does not exist: '%s'" % filePath)
        self.im   = Image.open(str(filePath))
    @property
    def size(self):
        return self.im.size
    @property
    def width(self):
        return self.im.size[0]
    @property
    def height(self):
        return self.im.size[1]
    @property
    def thinnest(self):
        "Return the minimum of the width and height"
        return min(self.width,self.height)
    @property
    def widest(self):
        "Return the maximum of the width and height"
        return max(self.width,self.height)
    def draw(self):
        gl.glBegin(gl.GL_QUADS)
        for (u,v) in ( (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), ):
            gl.glTexCoord2f(u, v)
            s, t = u-0.5, v-0.5
            gl.glVertex3f(s*self.widest, t*self.widest, 0.0)
        gl.glEnd()

# ------------------------------------------------------------------------------


DuotoneShaderFrag = '''
uniform sampler2D fillTex;
uniform vec4      fillTint;
void main() {
    vec2 st           = gl_TexCoord[0].st;
    vec4 fillC        = texture2D(fillTex,  st);
    float lum         = 0.29 * fillC[0] + 0.59 * fillC[1] + 0.12 * fillC[2];
    vec4  lumC        = vec4(lum,lum,lum,fillC[3]);
    //float cmix        = cos(2.0*3.1415926*lum) * 0.5 + 0.5;
    float cmix        = sin(3.1415926*lum);
    float hicon       = 0.5 - 0.5 * cos(3.1415926*pow(lum,0.15));
    float whites      = 0.5 - 0.5 * cos(3.1415926*pow(lum,3.0));
    gl_FragColor      = pow(lumC,0.35) * fillTint + whites;
    //gl_FragColor      = (1.0-cmix) * pow(lum,0.4) * fillTint + cmix * pow(lumC,1.5);
    //gl_FragColor      = vec4(cmix,cmix,cmix,1);
}'''




# ------------------------------------------------------------------------------




OverShaderFrag = '''
uniform sampler2D src1;
uniform sampler2D src2;
uniform float opacity;
void main() {
    vec4 base  = texture2D(src1, gl_TexCoord[0].st);
    vec4 other = texture2D(src2, gl_TexCoord[0].st);
    gl_FragColor = pow(1.0-opacity,0.454) * base + pow(opacity,0.454) * other;
}'''




# ------------------------------------------------------------------------------

LayeredShaderFrag = '''
uniform sampler2D fillTex;
uniform sampler2D matteTex;
uniform float matteBlur;
uniform float fillBlur;
uniform vec4  fillTint;
uniform float shadowBlur;
uniform vec2  shadowOffset;
uniform vec4  shadowTint;
uniform float hilightBlur;
uniform vec2  hilightOffset;
uniform vec4  hilightTint;
uniform float lolightBlur;
uniform vec2  lolightOffset;
uniform vec4  lolightTint;
uniform float coreBlur;
void main() {
    vec2 st           = gl_TexCoord[0].st;
    vec4 shadMatte    = texture2D(matteTex, st+shadowOffset, shadowBlur);
    vec4 hilightMatte = texture2D(matteTex, st+hilightOffset, hilightBlur);
    vec4 coreMatte    = texture2D(matteTex, st, coreBlur);
    vec4 lolightMatte = texture2D(matteTex, st+lolightOffset, lolightBlur);
    vec4 sharpMatte   = texture2D(matteTex, st, matteBlur);
    vec4 fillC        = texture2D(fillTex,  st, fillBlur);
    vec4 hilightC     = (1.0-hilightTint.w) * fillC * fillTint + hilightTint.w * hilightTint;
    vec4 lolightC     = (1.0-lolightTint.w) * fillC * fillTint + lolightTint.w * lolightTint;
    float hilightM    = hilightMatte.w * (1.0 - coreMatte.w);
    float lolightM    = lolightMatte.w * (1.0 - coreMatte.w);
    vec4 fgC          = (1.0-hilightM) * fillC * fillTint + hilightM * hilightC;
    fgC               = (1.0-lolightM) * fgC + lolightM * lolightC;
    gl_FragColor      = (1.0-sharpMatte.w) * shadowTint * shadMatte + sharpMatte.w * fgC;
}
'''


# ------------------------------------------------------------------------------


BlurShaderFrag = '''
uniform sampler2D image;
uniform float blur;
void main() {
	vec2 uv = gl_TexCoord[0].st;
	gl_FragColor = texture2D(image, uv, blur);
}'''

#--- GRADIENTS -----------------------------------------------------------------

LinearGradientFrag = '''
uniform vec4 clr1;
uniform vec4 clr2;
uniform float falloff;
uniform float slope;
void main() {
    float t    = pow(mix(gl_TexCoord[0].s, gl_TexCoord[0].t, slope), falloff);
    gl_FragColor = pow(1.0-t,2.2) * clr2 + pow(t,2.2) * clr1;
}'''

RadialGradientFrag = '''
uniform vec4 clr1;
uniform vec4 clr2;
void main() {
    float dx = gl_TexCoord[0].x - 0.5;
    float dy = gl_TexCoord[0].y - 0.5;
    float d = min(1.0, 4.0 * (dx*dx+dy*dy));
    gl_FragColor = pow(1.0-d,2.2) * clr2 + pow(d,2.2) * clr1;
}'''


# ------------------------------------------------------------------------------

GlassVert = '''
varying vec3  Normal;
varying vec3  EyeDir;

varying vec3  ReflectVec0;
varying vec3  ReflectVec1;
varying vec3  ReflectVec2;
varying vec3  ViewVec;

attribute vec3 tangent;
attribute vec3 binormal;
varying vec3 eyeVec;

void main(void)
{
    gl_Position    = ftransform();
    Normal         = normalize(gl_NormalMatrix * gl_Normal);
    vec4 pos       = gl_ModelViewMatrix * gl_Vertex;
    EyeDir         = pos.xyz;

    vec3 ecPos      = vec3 (gl_ModelViewMatrix * gl_Vertex);
    vec3 tnorm      = normalize(gl_NormalMatrix * gl_Normal);
    vec3 lightVec0   = normalize(gl_LightSource[0].position.xyz - ecPos);
    ReflectVec0      = normalize(reflect(-lightVec0, tnorm));
    vec3 lightVec1   = normalize(gl_LightSource[1].position.xyz - ecPos);
    ReflectVec1      = normalize(reflect(-lightVec1, tnorm));
    vec3 lightVec2   = normalize(gl_LightSource[2].position.xyz - ecPos);
    ReflectVec2      = normalize(reflect(-lightVec2, tnorm));
    ViewVec         = normalize(-ecPos);
    gl_Position     = ftransform();

    gl_TexCoord[0] = gl_MultiTexCoord0;

    mat3 TBN_Matrix;
    TBN_Matrix[0] =  gl_NormalMatrix * tangent;
    TBN_Matrix[1] =  gl_NormalMatrix * binormal;
    TBN_Matrix[2] =  gl_NormalMatrix * gl_Normal;
    vec4 Vertex_ModelView = gl_ModelViewMatrix * gl_Vertex;
    eyeVec = vec3(-Vertex_ModelView) * TBN_Matrix ;
}
'''

GlassFrag = '''
uniform float MixRatio;

uniform sampler2D EnvMap;
uniform sampler2D basetex;
uniform sampler2D bumptex;
uniform sampler2D spectex;

uniform float transparency;
uniform vec2 scaleBias;

varying vec3  Normal;
varying vec3  EyeDir;

varying vec3  ReflectVec0;
varying vec3  ReflectVec1;
varying vec3  ReflectVec2;
varying vec3  ViewVec;

varying vec3 eyeVec;

void main (void)
{
    const vec3 Xunitvec = vec3 (1.0, 0.0, 0.0);
    const vec3 Yunitvec = vec3 (0.0, 1.0, 0.0);
    vec3 reflectDir = reflect(EyeDir, Normal);
    vec2 index;

    vec2 texUV, srcUV = gl_TexCoord[0].xy;
    float height = texture2D(bumptex, srcUV).r;
    float v = height * scaleBias.x - scaleBias.y;
    vec3 eye = normalize(eyeVec);
    vec2 bmp = (eye.xy * v);
    texUV = srcUV + bmp;
    vec3 diff = texture2D(basetex, texUV).rgb;
    vec4 texBaseColor = vec4(diff, 1.0);
    float specmap = texture2D(spectex, texUV).r;


    index.y = dot(normalize(reflectDir), Yunitvec);
    reflectDir.y = 0.0;
    index.x = dot(normalize(reflectDir), Xunitvec) * 0.5;
    if (reflectDir.z >= 0.0)
        index = (index + 1.0) * 0.5;
    else
    {
        index.t = (index.t + 1.0) * 0.5;
        index.s = (-index.s) * 0.5 + 1.0;
    }
    vec3 e = vec3(texture2D(EnvMap, index));
    float em = dot(vec3(0.29, 0.59, 0.12), e);
    vec3 envColor = 1.30 * mix(e*2.0*pow(em,2.0), vec3(e), MixRatio);

    float facing = dot(Normal, ViewVec);

    vec3 nreflect0 = normalize(ReflectVec0);
    vec3 nreflect1 = normalize(ReflectVec1);
    vec3 nreflect2 = normalize(ReflectVec2);
    vec3 nview     = normalize(ViewVec+vec3(12.0*bmp*MixRatio,0.0));
    float spec0    = pow(max(dot(nreflect0, nview), 0.0), gl_FrontMaterial.shininess);
    float spec1    = pow(max(dot(nreflect1, nview), 0.0), gl_FrontMaterial.shininess);
    float spec2    = pow(max(dot(nreflect2, nview), 0.0), gl_FrontMaterial.shininess);

    vec3 spec = mix(1.0*envColor.xyz, (1.0-transparency)*texBaseColor.xyz, MixRatio) +
    	  vec3(gl_LightSource[0].specular) * vec3(spec0) +
    	  vec3(gl_LightSource[1].specular) * vec3(spec1) +
    	  vec3(gl_LightSource[2].specular) * vec3(spec2);
    vec3 color = mix(diff, spec, specmap);
    float lum = dot(vec3(0.29, 0.59, 0.12), color);
    gl_FragColor = vec4 (max(envColor, color),max(1.0-pow(facing,0.5),lum));
}
'''


# ------------------------------------------------------------------------------

archVert = '''
uniform sampler2D dsptex;
uniform float DspGain;
uniform float GradientSize;
uniform float BmpMix;

varying vec3  Normal;
varying vec3  ViewVec;

#define textureSize 256.0
#define texelSize 1.0 / textureSize

vec3 textureGradient2D( sampler2D tex, vec2 uv )
{
    vec2 f = fract( uv * textureSize );
    vec2 uvX = uv + vec2(GradientSize*texelSize, 0.0);
    vec2 uvY = uv + vec2(0.0, GradientSize*texelSize);
    float p  = texture2D(tex,uv).r;
    float px = texture2D(tex,uvX).r;
    float py = texture2D(tex,uvY).r;
    return vec3((p - px), 0.0, (p - py));
}

void main(void)
{
    Normal          = normalize(gl_NormalMatrix * gl_Normal);
    ViewVec         = normalize(-1.0 * vec3 (gl_ModelViewMatrix * gl_Vertex));
    gl_Position     = ftransform();
    gl_TexCoord[0]  = gl_MultiTexCoord0;
    vec2  srcUV     = gl_TexCoord[0].xy;
    vec3  dspTex    = Normal * (-1.0 + 2.0 * texture2D(dsptex, srcUV).r);
    vec3  grad      = textureGradient2D( dsptex, srcUV );
    Normal          = Normal + 1.0 * DspGain * grad;
    gl_Position     = gl_Position + DspGain * (1.0 - BmpMix) * vec4(dspTex,1.0);
}
'''

archFrag = '''
uniform sampler2D consttex;
uniform sampler2D basetex;
uniform sampler2D spectex;

uniform float DiffGain;
uniform float SpecGain;
uniform float SpecMapMix;
uniform float BmpMix;
uniform float ConstGain;

uniform float Opacity;

varying vec3  Normal;
varying vec3  ViewVec;

void main (void)
{
    vec3 LightVec0       = normalize(gl_LightSource[0].position.xyz + ViewVec);
    vec3 ReflectVec0     = normalize(reflect(-LightVec0, Normal));
    vec3 LightVec1       = normalize(gl_LightSource[1].position.xyz + ViewVec);
    vec3 ReflectVec1     = normalize(reflect(-LightVec1, Normal));
    vec3 LightVec2       = normalize(gl_LightSource[2].position.xyz + ViewVec);
    vec3 ReflectVec2     = normalize(reflect(-LightVec2, Normal));
    vec3 LightVec3       = normalize(gl_LightSource[3].position.xyz + ViewVec);
    vec3 ReflectVec3     = normalize(reflect(-LightVec3, Normal));
    
    float cross = pow(Opacity,1.0/2.2);

    vec2  srcUV    = gl_TexCoord[0].xy;
    
    vec3  consttexval   = texture2D(consttex, srcUV).rgb;
    vec3  diffTex  = texture2D(basetex, srcUV).rgb;
    float specmap  = 1.0 - texture2D(spectex, srcUV).r; // I don't know why.

    vec3  diff0    = vec3(gl_LightSource[0].diffuse) * pow(max(dot(Normal, LightVec0), 0.0),2.0);
    vec3  diff1    = vec3(gl_LightSource[1].diffuse) * pow(max(dot(Normal, LightVec1), 0.0),2.0);
    vec3  diff2    = vec3(gl_LightSource[2].diffuse) * pow(max(dot(Normal, LightVec2), 0.0),2.0);
    vec3  diff3    = vec3(gl_LightSource[3].diffuse) * pow(max(dot(Normal, LightVec3), 0.0),2.0);
    vec3  diff     = (diff0 + diff1 + diff2 + diff3) * diffTex;
    float spec0    = pow(max(dot(ReflectVec0, ViewVec), 0.0), gl_FrontMaterial.shininess);
    float spec1    = pow(max(dot(ReflectVec1, ViewVec), 0.0), gl_FrontMaterial.shininess);
    float spec2    = pow(max(dot(ReflectVec2, ViewVec), 0.0), gl_FrontMaterial.shininess);
    float spec3    = pow(max(dot(ReflectVec3, ViewVec), 0.0), gl_FrontMaterial.shininess);
    vec3 spec = mix(1.0,specmap,SpecMapMix) * (
          vec3(gl_LightSource[0].specular) * spec0 +
          vec3(gl_LightSource[1].specular) * spec1 +
          vec3(gl_LightSource[2].specular) * spec2 +
          vec3(gl_LightSource[3].specular) * spec3
          );
    gl_FragColor = vec4 (
        DiffGain * diff   * cross +
        SpecGain * spec   * cross +
        ConstGain * consttexval * cross,
        cross);
}
'''



# -------------------------------------------------------------------------------------

"""
Shader that uses two textures for the left and right eye.
Mode option selects:
    0:	left
    1:	right
    2:	anaglyph
    3:	scanline
Disparity arg sets the pixel offset at convergence point
"""

stereoVert = None

stereoFrag = '''
uniform sampler2D imageTex;
uniform sampler2D matteTex;
uniform float     depth;
uniform float     displayMode;
uniform float	  disparity;
void main() {
    vec2 st           = gl_TexCoord[0].st;
    vec4 imageC       = texture2D(imageTex,  st);
    gl_FragColor      = vec4(imageC[0],imageC[1],imageC[2],1);
}
'''

# -------------------------------------------------------------------------------------

depthLayerVert = """
uniform sampler2D DepthTex;
uniform float     DepthGain;
uniform float     DepthBias;

varying vec3  Normal;
varying vec3  ViewVec;

#define textureSize 256.0
#define texelSize 1.0 / textureSize

vec3 textureGradient2D( sampler2D tex, vec2 uv )
{
    vec2 f = fract( uv * textureSize );
    vec2 uvX = uv + vec2(texelSize, 0.0);
    vec2 uvY = uv + vec2(0.0, texelSize);
    float p  = texture2D(tex,uv).r;
    float px = texture2D(tex,uvX).r;
    float py = texture2D(tex,uvY).r;
    return vec3((p - px), 0.0, (p - py));
}

void main(void)
{
    Normal          = normalize(gl_NormalMatrix * gl_Normal);
    ViewVec         = normalize(-1.0 * vec3 (gl_ModelViewMatrix * gl_Vertex));
    gl_Position     = ftransform();
    gl_TexCoord[0]  = gl_MultiTexCoord0;
    vec2  srcUV     = gl_TexCoord[0].xy;
    vec3  dspTex    = Normal * (-1.0 + 2.0 * texture2D(DepthTex, srcUV).r);
    vec3 grad       = textureGradient2D( DepthTex, srcUV );
    Normal          = Normal + 10.0 * DepthGain * grad;
    gl_Position     = gl_Position + DepthGain * vec4(dspTex,1.0);
}
"""

depthLayerFrag = """
uniform sampler2D PlateTex;
uniform sampler2D MatteTex;
uniform float     MatteGain;
uniform float     UseMatte;
uniform float     showMatte;
uniform float     tintGain;
uniform vec3      tintColor;

void main (void)
{
    vec2  srcUV   = gl_TexCoord[0].xy;
    vec3  plate   = texture2D(PlateTex, srcUV).rgb;
    vec3  preComp;
    float alpha   = 1.0;
    vec3  matte;
    if (UseMatte != 0.0) {
        alpha     = MatteGain * texture2D(MatteTex, srcUV).r;
        float r   = alpha * pow( plate.r, 2.2);
        float g   = alpha * pow( plate.g, 2.2);
        float b   = alpha * pow( plate.b, 2.2);
        preComp   = vec3(pow(r,0.454), pow(g,0.454), pow(b,0.454));
    } else if (showMatte != 0.0) {
        matte     = texture2D(MatteTex, srcUV);
        preComp   = showMatte * matte + (1.0 - showMatte) * plate;
    } else {
        preComp   = plate;
    }
    preComp       = tintGain * tintColor + (1.0 - tintGain) * preComp;
    gl_FragColor  = vec4(preComp, alpha);
}
"""


exampleGeomVtx = '''
#version 120 
#extension GL_EXT_geometry_shader4 : enable

void main(void)
{
	//increment variable
	int i;
	for(i=0; i< gl_VerticesIn; i++){
		gl_Position = gl_PositionIn[i];
		EmitVertex();
	}
	EndPrimitive();																						
	for(i=0; i< gl_VerticesIn; i++){
		gl_Position = gl_PositionIn[i];
		gl_Position.xy = gl_Position.yx;
		EmitVertex();
	}
	EndPrimitive();	
																		      /////////////////////////////////////////////////////////////

}
'''

#-------------------------------------------------------------------------------

def _sphereCoordF(u,v):
    axisAng = u * math.pi * 2.0
    altAng = (2.0 * v - 1.0) * math.pi / 2.0
    y = math.sin(altAng)
    proj = math.cos(altAng)
    x = math.cos(axisAng) * proj
    z = math.sin(axisAng) * proj
    N = vec3(x,y,z).normalize()
    return N
    
def _sphereCoord(i, j, size=0.4, rows=96, cols=320):
    u = float(i) / float(cols-1)
    v = float(j) / float(rows-1)
    glTexCoord2d(1.0-u,v)
    N = _sphereCoordF(u,v)
    glNormal3d(N.x, N.y, N.z)
    P = N * size
    glVertex3d(P.x, P.y, P.z)


def _planeCoordF(u,v, center=vec3(0,0,0), xySpace=(vec3(1,0,0),vec3(0,1,0)), size=0.4):
    N = xySpace[0].cross(xySpace[1]).normalize()
    xy =   xySpace[0] * size * (2.0 * u - 1.0) + xySpace[1] * size * (2.0 * v - 1.0)
    P = xy + center
    return P, N

def _planeCoord(i, j, center=vec3(0,0,0), xySpace=(vec3(1,0,0),vec3(0,1,0)), size=0.4, rows=100, cols=100):
    v = float(j) / float(rows-1)
    u = float(i) / float(cols-1)
    glTexCoord2d(u,v)
    P, N = _planeCoordF(u,v, center, xySpace, size)
    glNormal3d(N.x, N.y, N.z)
    glVertex3d(P.x, P.y, P.z)


def _pieLatCoord(i, j, radius=0.4, rows=30, cols=8,
                center=vec3(0,0,0), 
                r1=0.0, r2=1.0, 
                lat1=50, long1=80, long2=120,
        ):
    u = float(i) / float(cols-1)
    v = float(j) / float(rows-1)
    glTexCoord2d(u,v)
    la1 = math.radians(lat1)  / math.pi 
    lo1 = math.radians(long1) / math.pi / 2.0
    lo2 = math.radians(long2) / math.pi / 2.0
    sphere1 = _sphereCoordF(lo1*u+lo2*(1.0-u), la1) * radius
    sphere2 = _sphereCoordF(lo1*u+lo2*(1.0-u) + 0.1, la1) * radius
    N = sphere1.cross(sphere2)
    P = sphere1 * (r1 * (1.0 - v) + r2 * v)
    glNormal3d(N.x, N.y, N.z)
    glVertex3d(P.x, P.y, P.z)

def _pieLongCoord(i, j, radius=0.4, rows=30, cols=8,
                center=vec3(0,0,0), 
                r1=0.0, r2=1.0, 
                lat1=50, lat2=80, long1=120,
        ):
    u = float(i) / float(cols-1)
    v = float(j) / float(rows-1)
    glTexCoord2d(u,v)
    la1 = math.radians(lat1)  / math.pi 
    la2 = math.radians(lat2)  / math.pi 
    lo1 = math.radians(long1) / math.pi / 2.0
    sphere1 = _sphereCoordF(lo1,la1*u+la2*(1.0-u)) * radius
    sphere2 = _sphereCoordF(lo1,la1*u+la2*(1.0-u) + 0.1) * radius
    N = sphere1.cross(sphere2)
    P = sphere1 * (r1 * (1.0 - v) + r2 * v)
    glNormal3d(N.x, N.y, N.z)
    glVertex3d(P.x, P.y, P.z)

def _pieEndCoord(i, j, radius=0.4, rows=30, cols=8,
                center=vec3(0,0,0), 
                r1=0.0,  
                lat1=50, lat2=80, long1=120, long2=120,
        ):
    u = float(i) / float(cols-1)
    v = float(j) / float(rows-1)
    la1 = math.radians(lat1)  / math.pi 
    la2 = math.radians(lat2)  / math.pi 
    lo1 = math.radians(long1) / math.pi / 2.0
    lo2 = math.radians(long2) / math.pi / 2.0
    glTexCoord2d(1.0 - ((1-u)*lo1+u*lo2), (1-v)*la1+v*la2)
    #glTexCoord2d(1.0 - ((1-v)*la1+v*la2), (1-u)*lo1+u*lo2)
    #N = _sphereCoordF(lo1*v+lo2*(1.0-v), la1*u+la2*(1.0-u))
    N = _sphereCoordF((1-u)*lo1+u*lo2, (1-v)*la1+v*la2)
    P = N * radius * r1
    glNormal3d(N.x, N.y, N.z)
    glVertex3d(P.x, P.y, P.z)



#-------------------------------------------------------------------------------

class _TestWindow(QGLShaderWidget):

    #clicked = QtCore.pyqtSignal()
    fontB18 = QtGui.QFont("Helvetica", 18, QtGui.QFont.Bold)

    def __init__(self, vtxSrc, fragSrc):
        super(_TestWindow, self).__init__()

        self.sectionObject = None
        self.worldObject = None
        
        self.cross = 1.0
        self.crossDir = -1.0
        self.crossSpeed = 0.05
        self.crossTimer = None

        self.clearColor = QtGui.QColor(8,8,8)
        self.xRot = 0
        self.yRot = 180 * 16.0
        self.zRot = 0
        
        self.cam = vec3(0.1,0.1,1.2)

        self.lastPos = QtCore.QPoint()
        self.throw = (1,0)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.timerCB)
        timer.start(20)

        self.setWindowTitle("QGLShaderWidget")

        self.defineShader("crust",   vtxSrc, fragSrc)
        self.defineShader("section", vtxSrc, fragSrc)
        self.defineShader("world",   vtxSrc, fragSrc)
        
        self.fontB12 = QtGui.QFont("Helvetica", 12, QtGui.QFont.Bold)

    def startCrossCB(self):
        if self.crossTimer:
            self.crossTimer.stop()
        if self.cross < 0.5:
            self.crossDir = 1.0
        else:
            self.crossDir = -1.0
        self.crossTimer = QtCore.QTimer(self)
        self.crossTimer.timeout.connect(self.crossCB)
        self.crossTimer.start(20)

    def stopCrossCB(self):
        self.crossTimer.stop()
        
    def crossCB(self):
        self.cross += self.crossSpeed * self.crossDir
        if self.cross < 0.0:
            self.crossDir = 1.0
            self.cross = 0.0
            self.crossTimer.stop()
        elif self.cross > 1.0:
            self.crossDir = -1.0
            self.cross = 1.0
            self.crossTimer.stop()

    def keyPressEvent(self, event):
        #if event.modifiers() & QtCore.Qt.AltModifier:
        if event.key() in (QtCore.Qt.Key_A,):
            self.startCrossCB()
            event.accept()

    def timerCB(self):
        mouse_state = app.mouseButtons()
        if mouse_state == QtCore.Qt.NoButton:
            self.rotateBy(self.throw[1], self.throw[0], 0)

    def rotateBy(self, xAngle, yAngle, zAngle):
        self.xRot += xAngle
        self.yRot += yAngle
        self.zRot += zAngle
        self.updateGL()

    def minimumSizeHint(self):
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        return QtCore.QSize(800, 800)

    def setClearColor(self, color):
        self.clearColor = color
        self.updateGL()

    def makeGeometry(self):
        self.sectionObject = self.makeSlice(radius1=0.0005, radius2=0.44, center=vec3(0,0,0), 
            lat1=0, lat2=180, long1=180, long2=360, 
            rows=20, cols=20)
        self.crustObject = self.makeSlice(radius1=0.44, radius2=0.45, center=vec3(0,0,0), 
            lat1=120, lat2=150, long1=270, long2=310, 
            rows=20, cols=20)
        self.crustSecObject = self.makeSlice(radius1=0.0005, radius2=0.44, center=vec3(0,0,0), 
            lat1=120, lat2=150, long1=270, long2=310, 
            rows=20, cols=20)
        self.worldObject = self.makeSlice(radius1=0.44, radius2=0.45, center=vec3(0,0,0), 
            lat1=0, lat2=180, long1=180, long2=360, 
            rows=90, cols=120)
        #self.worldObject = self.makeSphere()
            
    def initializeGL(self):
        #super(TestWindow, self).initializeGL()
        
        if not self.sectionObject:
            self.makeGeometry()
        
        glEnable(GL_DEPTH_TEST)
        glClearDepth(1.0)

        #glEnable(GL_CULL_FACE)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
#-------------------------------------------------------------------------------
        #glPatchParameteri(GL_PATCH_VERTICES, 16)
        #glDrawArrays(GL_PATCHES, firstVert, vertCount)
#-------------------------------------------------------------------------------


    def paintGL(self):

        self.qglClearColor(self.clearColor)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        
        glPushMatrix()
        glLoadIdentity()
        glTranslated(-self.cam.x,-self.cam.y,-self.cam.z)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_LIGHT2)
        glEnable(GL_LIGHT3)

        glLightfv(GL_LIGHT0, GL_DIFFUSE,  vec4(0.9,0.9,0.85,1.0).asTuple())
        glLightfv(GL_LIGHT0, GL_SPECULAR, vec4(0.9,0.9,0.85,1.0).asTuple())
        glLightfv(GL_LIGHT0, GL_POSITION, vec4(-15.5,3.5,10.5,1.0).asTuple())

        glLightfv(GL_LIGHT1, GL_DIFFUSE,  vec4(0.15,0.2,0.2,1.0).asTuple())
        glLightfv(GL_LIGHT1, GL_SPECULAR, vec4(0.15,0.2,0.2,1.0).asTuple())
        glLightfv(GL_LIGHT1, GL_POSITION, vec4(15.5,1.5,-5.5,1.0).asTuple())

        glLightfv(GL_LIGHT2, GL_DIFFUSE,  vec4(0.3,0.3,0.35,1.0).asTuple())
        glLightfv(GL_LIGHT2, GL_SPECULAR, vec4(0.3,0.3,0.35,1.0).asTuple())
        glLightfv(GL_LIGHT2, GL_POSITION, vec4(0.5,15.5,-7.5,1.0).asTuple())

        glLightfv(GL_LIGHT3, GL_DIFFUSE,  vec4(0.1,0.1,0.2,1.0).asTuple())
        glLightfv(GL_LIGHT3, GL_SPECULAR, vec4(0.1,0.1,0.2,1.0).asTuple())
        glLightfv(GL_LIGHT3, GL_POSITION, vec4(0.5,-15.5,-5.5,1.0).asTuple())
        
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec4(1, 1, 1, 1.0).asTuple())
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec4(1, 1, 1, 1).asTuple())

        glMaterialf(GL_FRONT_AND_BACK,  GL_SHININESS, 20.0)

        glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
        glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0)
        
        glEnable(GL_DEPTH_TEST)
        
        glColor3f(1.0,1.0,0.0)
        self['crust']['Opacity'] = 1.0 - self.cross
        self['world']['Opacity'] = self.cross
        self['section']['Opacity'] = 1.0
        self['section']['ConstGain'] = 1.0
        
        #if (1.0-self.cross) < 0.5:
        #    glDisable(GL_DEPTH_TEST)
        #else:
        #    glEnable(GL_DEPTH_TEST)
        
        if (1.0 - self.cross) > 0.001:
            self.applyShader("crust")
            glCallList(self.crustObject)
            self.removeShader()
            
            self.applyShader("section")
            glCallList(self.crustSecObject)
            self.removeShader()

        #if self.cross < 0.5:
        #    glDisable(GL_DEPTH_TEST)
        #else:
        #    glEnable(GL_DEPTH_TEST)
        
        if self.cross > 0.001:
            self.applyShader("world")
            glCallList(self.worldObject)
            self.removeShader()
    
            self.applyShader("section")
            glCallList(self.sectionObject)
            self.removeShader()

        glPopMatrix()

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40., width / float(height), 0.3, 3.0)
        glMatrixMode(GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        if event.buttons() & QtCore.Qt.LeftButton:
            self.rotateBy(8 * dy, 8 * dx, 0)
        elif event.buttons() & QtCore.Qt.RightButton:
            self.rotateBy(8 * dy, 0, 8 * dx)
        self.lastPos = event.pos()
        self.throw = (dx,dy)

    #def mouseReleaseEvent(self, event):
        #self.clicked.emit()        

    def makeSlice(self, radius1=0.0005, radius2=0.45, center=vec3(0,0,0), 
                lat1=120, lat2=150, long1=270, long2=310, 
                rows=20, cols=20):
        
        dlist = glGenLists(1)
        glNewList(dlist, GL_COMPILE)

        if radius1 > 0.001:
            for i in range(cols-1):
                glBegin(GL_TRIANGLE_STRIP)
                _pieEndCoord(i, 0, 1.0, rows, cols, center, radius1, lat1, lat2, long1, long2)
                _pieEndCoord(i+1, 0, 1.0, rows, cols, center, radius1, lat1, lat2, long1, long2)
                for j in range(rows-1):
                    _pieEndCoord(i, j+1, 1.0, rows, cols, center, radius1, lat1, lat2, long1, long2)
                    _pieEndCoord(i+1,j+1, 1.0, rows, cols, center, radius1, lat1, lat2, long1, long2)
                glEnd()

        if radius2 > 0.001:
            for j in range(rows-1):
                glBegin(GL_TRIANGLE_STRIP)
                _pieEndCoord(0, j,   1.0, rows, cols, center, radius2, lat1, lat2, long1, long2)
                _pieEndCoord(0, j+1, 1.0, rows, cols, center, radius2, lat1, lat2, long1, long2)
                for i in range(cols-1):
                    _pieEndCoord(i+1, j,   1.0, rows, cols, center, radius2, lat1, lat2, long1, long2)
                    _pieEndCoord(i+1, j+1, 1.0, rows, cols, center, radius2, lat1, lat2, long1, long2)
                glEnd()



        for i in range(cols-1):
            glBegin(GL_TRIANGLE_STRIP)
            _pieLatCoord(i, 0, 1.0, rows, cols, center, radius1, radius2, lat1, long1, long2)
            _pieLatCoord(i+1, 0, 1.0, rows, cols, center, radius1, radius2, lat1, long1, long2)
            for j in range(rows-1):
                _pieLatCoord(i, j+1, 1.0, rows, cols, center, radius1, radius2, lat1, long1, long2)
                _pieLatCoord(i+1,j+1, 1.0, rows, cols, center, radius1, radius2, lat1, long1, long2)
            glEnd()

        for i in range(cols-1):
            glBegin(GL_TRIANGLE_STRIP)
            _pieLatCoord(i, 0, 1.0, rows, cols, center, radius1, radius2, lat2, long1, long2)
            _pieLatCoord(i+1, 0, 1.0, rows, cols, center, radius1, radius2, lat2, long1, long2)
            for j in range(rows-1):
                _pieLatCoord(i, j+1, 1.0, rows, cols, center, radius1, radius2, lat2, long1, long2)
                _pieLatCoord(i+1,j+1, 1.0, rows, cols, center, radius1, radius2, lat2, long1, long2)
            glEnd()

        for i in range(cols-1):
            glBegin(GL_TRIANGLE_STRIP)
            _pieLongCoord(i, 0, 1.0, rows, cols, center, radius1, radius2, lat1, lat2, long1)
            _pieLongCoord(i+1, 0, 1.0, rows, cols, center, radius1, radius2, lat1, lat2, long1)
            for j in range(rows-1):
                _pieLongCoord(i, j+1, 1.0, rows, cols, center, radius1, radius2, lat1, lat2, long1)
                _pieLongCoord(i+1,j+1, 1.0, rows, cols, center, radius1, radius2, lat1, lat2, long1)
            glEnd()

        for i in range(cols-1):
            glBegin(GL_TRIANGLE_STRIP)
            _pieLongCoord(i, 0, 1.0, rows, cols, center, radius1, radius2, lat1, lat2, long2)
            _pieLongCoord(i+1, 0, 1.0, rows, cols, center, radius1, radius2, lat1, lat2, long2)
            for j in range(rows-1):
                _pieLongCoord(i, j+1, 1.0, rows, cols, center, radius1, radius2, lat1, lat2, long2)
                _pieLongCoord(i+1,j+1, 1.0, rows, cols, center, radius1, radius2, lat1, lat2, long2)
            glEnd()



        glEndList()
        return dlist

    def makePlane(self, size=0.4, rows=20, cols=20, center=vec3(0,0,0), 
            xVector=vec3(1,0,0), yVector=vec3(0,1,0)):
        xy = (xVector,yVector)
        dlist = glGenLists(1)
        glNewList(dlist, GL_COMPILE)

        for i in range(cols-1):
            glBegin(GL_TRIANGLE_STRIP)
            _planeCoord(i,0, center, xy, size,rows,cols)
            _planeCoord(i+1,0, center, xy, size,rows,cols)
            for j in range(rows-1):
                _planeCoord(i,j+1, center, xy, size,rows,cols)
                _planeCoord(i+1,j+1, center, xy, size,rows,cols)
            glEnd()

        glEndList()
        return dlist

    def makeCube(self, size=0.4, rows=100, cols=100):
        planes = [
        self.makePlane(size, rows, cols, 
            center=vec3(0,0,size), xVector=vec3(1,0,0), yVector=vec3(0,1,0)),
        self.makePlane(size, rows, cols, 
            center=vec3(0,size,0), xVector=vec3(1,0,0), yVector=vec3(0,0,-1)),
        self.makePlane(size, rows, cols, 
            center=vec3(size,0,0), xVector=vec3(0,0,-1), yVector=vec3(0,1,0)),
        self.makePlane(size, rows, cols, 
            center=vec3(0,0,-size), xVector=vec3(-1,0,0), yVector=vec3(0,1,0)),
        self.makePlane(size, rows, cols, 
            center=vec3(0,-size,0), xVector=vec3(1,0,0), yVector=vec3(0,0,1)),
        self.makePlane(size, rows, cols, 
            center=vec3(-size,0,0), xVector=vec3(0,0,1), yVector=vec3(0,1,0)),
        ]
        dlist = glGenLists(1)
        glNewList(dlist, GL_COMPILE)
        for i in planes:
            glCallList(i)
        glEndList()
        return dlist
        

    def makeCubeX(self,size=0.4, rows=200, cols=200):
        coords = (
            ( ( +1, -1, -1 ), ( -1, -1, -1 ), ( -1, +1, -1 ), ( +1, +1, -1 ) ),
            ( ( +1, +1, -1 ), ( -1, +1, -1 ), ( -1, +1, +1 ), ( +1, +1, +1 ) ),
            ( ( +1, -1, +1 ), ( +1, -1, -1 ), ( +1, +1, -1 ), ( +1, +1, +1 ) ),
            ( ( -1, -1, -1 ), ( -1, -1, +1 ), ( -1, +1, +1 ), ( -1, +1, -1 ) ),
            ( ( +1, -1, +1 ), ( -1, -1, +1 ), ( -1, -1, -1 ), ( +1, -1, -1 ) ),
            ( ( -1, -1, +1 ), ( +1, -1, +1 ), ( +1, +1, +1 ), ( -1, +1, +1 ) )
        )
        dlist = glGenLists(1)
        glNewList(dlist, GL_COMPILE)
        for i in range(6):
            glBegin(GL_QUADS)
            for j in range(4):
                tx = {False: 0, True: 1}[j == 0 or j == 3]
                ty = {False: 0, True: 1}[j == 0 or j == 1]
                glTexCoord2d(tx, ty)
                glVertex3d(size * coords[i][j][0],
                           size * coords[i][j][1],
                           size * coords[i][j][2])
            glEnd()
        glEndList()
        return dlist

    def makeSphere(self, size=0.45, rows=75, cols=280):
                
        dlist = glGenLists(1)
        glNewList(dlist, GL_COMPILE)
        
        for i in range(cols-1):
            glBegin(GL_TRIANGLE_STRIP)
            _sphereCoord(i,1,size,rows,cols)
            _sphereCoord(i+1,1,size,rows,cols)
            for j in range(2,rows-2):
                _sphereCoord(i,j+1,size,rows,cols)
                _sphereCoord(i+1,j+1,size,rows,cols)
            glEnd()
        
        glBegin(GL_TRIANGLE_FAN)
        _sphereCoord(0,0,size,rows,cols)
        for i in range(cols):
            _sphereCoord(i,1,size,rows,cols)
        glEnd()
    
        glBegin(GL_TRIANGLE_FAN)
        _sphereCoord(0,rows-1,size,rows,cols)
        for i in range(cols):
            _sphereCoord(i,rows,size,rows,cols)
        glEnd()
        
        glEndList()
        return dlist



# ------------------------------------------------------------------------------

if __name__ == '__main__':


    import sys
    #resourceImageDir = Path(r'C:\Python26\Lib\site-packages')
    resourceImageDir = Path(__file__).dirname() / "textures"
    tectonicDir = Path(r"E:\Work\Code\tectonic\images\Paleogeographic")

    #QtOpenGL.QGL.setPreferredPaintEngine(QtGui.QPaintEngine.OpenGL)
    app = QtGui.QApplication(sys.argv)
    #print "Qt version:",(QtCore.qVersion())
    
    if True:
        
        window = _TestWindow(archVert,archFrag)
        
        window['crust']['dsptex']       = tectonicDir / 'maps_dsp.000.png'
        window['crust']['basetex']      = tectonicDir / 'maps_diff.000.png'
        window['crust']['spectex']      = tectonicDir / 'maps_spec.000.png'
        window['crust']['consttex']       = tectonicDir / 'crosssect1_irr.png'
        window['crust']['DspGain']      = 0.005
        window['crust']['GradientSize'] = 0.25
        window['crust']['DiffGain']     = 0.7
        window['crust']['SpecGain']     = 0.4
        window['crust']['SpecMapMix']   = 1.0
        window['crust']['Opacity']      = 1.0
        window['crust']['BmpMix']       = 0.5
        window['crust']['ConstGain']      = 0.0
        
        window['world']['dsptex']       = tectonicDir / 'maps_dsp.000.png'
        window['world']['basetex']      = tectonicDir / 'maps_diff.000.png'
        window['world']['spectex']      = tectonicDir / 'maps_spec.000.png'
        window['world']['consttex']       = tectonicDir / 'crosssect1_irr.png'
        window['world']['DspGain']      = 0.025
        window['world']['GradientSize'] = 0.75
        window['world']['DiffGain']     = 0.9
        window['world']['SpecGain']     = 0.8
        window['world']['SpecMapMix']   = 1.0
        window['world']['Opacity']      = 1.0
        window['world']['BmpMix']       = 0.5
        window['world']['ConstGain']      = 0.0

        #window['section']['dsptex']       = tectonicDir / 'crosssect1_bmp.png'
        #window['section']['basetex']      = tectonicDir / 'crosssect1_diff.png'
        #window['section']['spectex']      = tectonicDir / 'maps_spec.000.png'
        window['section']['consttex']       = tectonicDir / 'crosssect1_irr.png'
        #window['section']['DspGain']      = 0.015
        #window['section']['GradientSize'] = 0.75
        window['section']['DiffGain']     = 0.1
        window['section']['SpecGain']     = 0.0
        window['section']['SpecMapMix']   = 0.0
        window['section']['Opacity']      = 1.0
        window['section']['BmpMix']       = 1.0
        window['section']['ConstGain']      = 0.75
    
    if False:
        window = _TestWindow(GlassVert,GlassFrag)
        window['default']['EnvMap']       = resourceImageDir / 'reflectionEnv.png'
        window['default']['basetex']      = resourceImageDir / 'diffuse.png'
        window['default']['bumptex']      = resourceImageDir / 'displacement.png'
        window['default']['spectex']      = resourceImageDir / 'specular.png'
        window['default']['transparency'] = 0.9
        window['default']['scaleBias']    = (1.0,1.0)
        window['default']['MixRatio']     = 0.0
        
    if False:
        window = _TestWindow(None,BlurShaderFrag)
        window['default']['image'] = resourceImageDir / 'diffuse.png'
        window['default']['blur'] = 2.0
        
    if False:
        window = _TestWindow(RadialGradientFrag)
        window['default']['clr1'] = (1.0,0.0,0.0,1.0)
        window['default']['clr2'] = (0.0,1.0,1.0,0.3)
        
    window.show()
    sys.exit(app.exec_())








