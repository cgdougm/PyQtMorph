#!/bin/env python26

"""
doug
08-26-11
"""
import sys, os
import math
from lxml.etree import parse, Element, ElementTree

from   PyQt4.QtGui       import *
from   PyQt4.QtCore      import *
from   PyQt4.QtOpenGL    import *

from   OpenGL.GL         import *
from   OpenGL.GLU        import *

sys.path.append("python")

from path import path as Path

from qtshader import QGLShaderWidget
from toolenv  import ToolEnv
from triangle import TriCanvas, TriSample


FILEVERSION = "1.0"

MAXRES  = (1600,900) # TODO: get this automatically

def clamp(x,lo,hi): return min(max(x,lo),hi)


def clipToUnit(x0,y0,x1,y1):
    """
    Clip line segment (x0,y0) to (x1,y1) inside the unit square.
    The endpoint (x0,y0) is known to be inside;
    return clipped (x1,y1) to the edges.
    """
    if x0 < 0.0 or x0 > 1.0 or y0 < 0.0 or y0 > 1.0:
        raise Exception("(x0,y0) must be in unit square (got [%s,%s])" % (x0,y0))
    if x1 < 0.0: # X crosses y=0 axis
        t = x0 / abs(x1-x0)
        xc = 0.0
        yc = y0 + (y1-y0) * t
    elif x1 > 1.0: # X crosses y=1 axis
        t = (1.0-x0) / abs(x1-x0)
        xc = 1.0
        yc = y0 + (y1-y0) * t
    else: # X not needing clipping
        xc = x1
        yc = y1
    if yc < 0.0: # Y crosses x=0 axis
        t = y0 / abs(yc-y0)
        yc = 0.0
        xc = x0 + (xc-x0) * t
    elif yc > 1.0: # Y crosses x=1 axis
        t = (1.0-y0) / abs(yc-y0)
        yc = 1.0
        xc = x0 + (xc-x0) * t
    else: # X not needing clipping
        pass
    return xc, yc

from cgkit.cgtypes import vec3, vec4

try:
    from PIL import Image
except ImportError:
    import Image # linux

def getImageSize(imgPath):
    try:
        im = Image.open(imgPath)
    except:
        return None
    return im.size

# ------------------------------------------------------------------------------
# Start app

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('plastique')) #'cleanlooks'))

# ------------------------------------------------------------------------------
# Get OpenGL

try:
    from OpenGL import GL
except ImportError:
    print "No GL"
    sys.exit(-1)
    app = QApplication(sys.argv)
    QMessageBox.critical(None, "OpenGL hellogl",
                            "PyOpenGL must be installed to run this example.",
                            QMessageBox.Ok | QMessageBox.Default,
                            QMessageBox.NoButton)
    sys.exit(1)

if __name__ == "__main__":
    E = ToolEnv(app)

# ------------------------------------------------------------------------------
#

class RenderDialog(QDialog):
    """
    """
    
    def __init__(self,parent=None):
        super(RenderDialog,self).__init__(parent)

        self.setWindowTitle('Render')
        self.setModal(True)

        self.mainLayout = QGridLayout()
        self.setLayout(self.mainLayout)


        row = 0  # ----------------------------------------
        self.dirLabelB = QPushButton("Output Directory")
        self.mainLayout.addWidget(self.dirLabelB, row, 0)
        self.connect(self.dirLabelB,    SIGNAL('clicked()'), self.setDirectoryCB)
        
        self.dirL = QLabel("")
        self.mainLayout.addWidget(self.dirL, row, 1, 1, 2)

        row += 1 # ----------------------------------------
        self.fileLabelL = QLabel("Output Filename")
        self.mainLayout.addWidget(self.fileLabelL, row, 0)
        
        self.fileL = QLineEdit("",self)
        self.mainLayout.addWidget(self.fileL, row, 1, 1, 2)
        self.connect(self.fileL, SIGNAL("textChanged (const QString&)"), self.checkReady)

        row += 1 # ----------------------------------------
        self.frameL = QLabel("Frame Start/End")
        self.mainLayout.addWidget(self.frameL, row, 0)
        
        self.startFrameSB = QSpinBox(self)
        self.mainLayout.addWidget(self.startFrameSB, row, 1)
        self.startFrameSB.setRange(1,10000)

        self.endFrameSB = QSpinBox(self)
        self.mainLayout.addWidget(self.endFrameSB, row, 2)
        self.endFrameSB.setRange(1,10000)

        row += 1 # ----------------------------------------
        self.resL = QLabel("Res width/height")
        self.mainLayout.addWidget(self.resL, row, 0)
        
        self.widthSB = QSpinBox(self)
        self.widthSB.setRange(500,3000)
        self.widthSB.setSingleStep(500)
        self.mainLayout.addWidget(self.widthSB, row, 1)

        self.heightSB = QSpinBox(self)
        self.heightSB.setRange(500,3000)
        self.heightSB.setSingleStep(500)
        self.mainLayout.addWidget(self.heightSB, row, 2)

        row += 1 # ----------------------------------------
        self.renderB    = QPushButton('Render')
        self.connect(self.renderB,    SIGNAL('clicked()'), self.renderCB)
        self.mainLayout.addWidget(self.renderB, row, 1)
        self.renderB.setEnabled(False)
        
        self.cancelB = QPushButton('Cancel')
        self.connect(self.cancelB, SIGNAL('clicked()'), self.reject)
        self.mainLayout.addWidget(self.cancelB, row, 2)

    def checkReady(self,*args):
        d = Path(str(self.dirL.text()))
        f = str(self.fileL.text()).strip()
        s = self.startFrameSB.value()
        e = self.endFrameSB.value()
        ready = d.isdir() and len(f)>0 and s < e
        #print "READY",ready,d.isdir(), len(f)>0, s < e
        self.renderB.setEnabled(ready)

    def clear(self):
        self.recipientLE.clear()
        self.bodyTE.clear()
        self.bodyTE.setFont(QFont("Courier",9))
        self.checkReady()

    def setup(self):
        self.startFrameSB.setValue(1)
        self.endFrameSB.setValue(30)
        self.widthSB.setValue(1000)
        self.heightSB.setValue(500)
        self.checkReady()

    def setDirectoryCB(self):
        defaultDir = Path().getcwd().abspath()
        d = QFileDialog.getExistingDirectory(self, 
                    "Save to directory",
                    str(defaultDir),
                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if d:
            self.dirL.setText(str(d))
            self.checkReady()

    def renderCB(self):
        d = Path(str(self.dirL.text()))
        f = str(self.fileL.text()).strip()
        s = self.startFrameSB.value()
        e = self.endFrameSB.value()
        w = self.widthSB.value()
        h = self.heightSB.value()
        self.parent().doRenderSequence(d,f,s,e,w,h)
        self.accept()

# ------------------------------------------------------------------------------
#

class SettingsDialog(QDialog):
    """
    """
    
    def __init__(self,parent=None):
        super(SettingsDialog,self).__init__(parent)
        
        self.recentDir = Path().getcwd()

        self.setWindowTitle('Settings')
        self.setModal(True)

        self.mainLayout = QGridLayout()
        self.setLayout(self.mainLayout)

        row = 0  # ----------------------------------------
        self.aLabelB = QPushButton("Image A")
        self.mainLayout.addWidget(self.aLabelB, row, 0)
        self.connect(self.aLabelB,    SIGNAL('clicked()'), self.setImageACB)
        self.aL = QLabel("")
        self.mainLayout.addWidget(self.aL, row, 1, 1, 2)

        row += 1 # ----------------------------------------
        self.bLabelB = QPushButton("Image B")
        self.mainLayout.addWidget(self.bLabelB, row, 0)
        self.connect(self.bLabelB,    SIGNAL('clicked()'), self.setImageBCB)
        self.bL = QLabel("")
        self.mainLayout.addWidget(self.bL, row, 1, 1, 2)

        row += 1 # ----------------------------------------
        self.okayB    = QPushButton('Okay')
        self.connect(self.okayB,    SIGNAL('clicked()'), self.okayCB)
        self.mainLayout.addWidget(self.okayB, row, 1)
        
        self.cancelB = QPushButton('Cancel')
        self.connect(self.cancelB, SIGNAL('clicked()'), self.reject)
        self.mainLayout.addWidget(self.cancelB, row, 2)

    def clear(self):
        self.aL.clear()
        self.bL.clear()

    def setup(self,a,b):
        self.recentDir = Path(str(a)).dirname()
        self.aL.setText(str(a))
        self.bL.setText(str(b))

    def setImageACB(self):
        defaultDir = self.recentDir
        f = QFileDialog.getOpenFileName (self, 
                    "Open 'A' image",
                    str(defaultDir),
                    "Images (*.png *.jpg)", 
                    )
        if f:
            self.aL.setText(str(f))
            self.recentDir = Path(str(f)).dirname()

    def setImageBCB(self):
        defaultDir = self.recentDir
        f = QFileDialog.getOpenFileName (self, 
                    "Open 'B' image",
                    str(defaultDir),
                    "Images (*.png *.jpg)", 
                    )
        if f:
            self.bL.setText(str(f))
            self.recentDir = Path(str(f)).dirname()

    def okayCB(self):
        a = Path(str(self.aL.text()).strip())
        b = Path(str(self.bL.text()).strip())
        self.recentDir = a.dirname()
        self.parent().setImages(a,b)
        self.accept()


# ------------------------------------------------------------------------------
#

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        

        self.setWindowTitle('Morph')
        #self.setWindowIcon(E.getIcon("logo"))

        # Central widget
        self.centralW = MorphWidget(self)
        self.setCentralWidget(self.centralW)

        # Tools
        self.makeMenus()
        self.makeToolBar()
        self.makeToolBoxDW()
        
        # Dialogs
        self.renderD   = RenderDialog(self)
        self.settingsD = SettingsDialog(self)

        #self.initializeUI()
        QTimer.singleShot(200, self.initializeUi)

        self.statusBar().showMessage('Ready',2000)
        
        self.latestSaveFile = None

    def makeMenus(self):
        menubar = self.menuBar()
        
        file = menubar.addMenu('&Application')
        
        newFile = QAction(E.getIcon("New"), 'New...', self)
        newFile.setShortcut('Ctrl+N')
        newFile.setStatusTip('Start a new Morph with settings...')
        self.connect(newFile, SIGNAL('triggered()'), self.newCB)
        file.addAction(newFile)
        
        openFile = QAction(E.getIcon("Folder"), 'Open...', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open Morph XML data file')
        self.connect(openFile, SIGNAL('triggered()'), self.openFileCB)
        file.addAction(openFile)
        
        saveFile = QAction(E.getIcon("save"), 'Save...', self)
        saveFile.setShortcut('Ctrl+S')
        saveFile.setStatusTip('Save Morph XML data file')
        self.connect(saveFile, SIGNAL('triggered()'), self.saveFileCB)
        file.addAction(saveFile)
        
        file.addSeparator()

        settings = QAction(E.getIcon("gear"), 'Settings...', self)
        settings.setShortcut('Ctrl+S')
        settings.setStatusTip('Open Morhph settings window')
        self.connect(settings, SIGNAL('triggered()'), self.settingsDialogRequestCB)
        file.addAction(settings)
        
        file.addSeparator()

        export = QAction(E.getIcon("Picture-Save"), 'Export image...', self)
        export.setShortcut('Ctrl+E')
        export.setStatusTip('Render current image to file')
        self.connect(export, SIGNAL('triggered()'), self.exportImageCB)
        file.addAction(export)
        
        render = QAction(E.getIcon("Film-Save"), 'Render...', self)
        render.setShortcut('Ctrl+R')
        render.setStatusTip('Render image sequence dialog')
        self.connect(render, SIGNAL('triggered()'), self.renderDialogRequestCB)
        file.addAction(render)
        
        file.addSeparator()

        exit = QAction('Exit', self)
        exit.setShortcut('Ctrl+Q')
        exit.setStatusTip('Exit application')
        self.connect(exit, SIGNAL('triggered()'), SLOT('close()'))
        file.addAction(exit)

        edit = menubar.addMenu("&Edit")
        
        clearEditAction = QAction(E.getIcon("Bomb"), 'Clear', self)
        edit.addAction(clearEditAction)
        self.connect(clearEditAction,SIGNAL("triggered (bool)"), self.clearCB)

        view = menubar.addMenu("&View")
        
        toolboxViewAction = QAction(E.getIcon("Setting-Tools"), 'Toolbox', self)
        view.addAction(toolboxViewAction)
        def toolboxViewActionCB(b):
            self.toolboxDW.show()
        self.connect(toolboxViewAction,SIGNAL("triggered (bool)"),toolboxViewActionCB)

        view.addSeparator()
        
        trianglesViewAction = QAction(E.getIcon("Draw-Polyline"), 'Triangles', self)
        view.addAction(trianglesViewAction)
        self.connect(trianglesViewAction,SIGNAL("triggered (bool)"),self.toggleTriangleDisplayCB)

        vectorsViewAction = QAction(E.getIcon("Draw-Line"), 'Vectors', self)
        view.addAction(vectorsViewAction)
        self.connect(vectorsViewAction,SIGNAL("triggered (bool)"),self.toggleVectorDisplayCB)

        pointsViewAction = QAction(E.getIcon("Draw-Vertex"), 'Points', self)
        view.addAction(pointsViewAction)
        self.connect(pointsViewAction,SIGNAL("triggered (bool)"),self.togglePointDisplayCB)
        
        view.addSeparator()
        
        resetViewAction = QAction(E.getIcon("Map"), 'Reset view', self)
        view.addAction(resetViewAction)
        self.connect(resetViewAction,SIGNAL("triggered (bool)"),self.resetViewCB)


    def toggleTriangleDisplayCB(self):
        self.centralW.triangleDisplay  = 1 - self.centralW.triangleDisplay
        
    def toggleVectorDisplayCB(self):
        self.centralW.vectorDisplay    = 1 - self.centralW.vectorDisplay
        
    def togglePointDisplayCB(self):
        self.centralW.pointDisplay     = 1 - self.centralW.pointDisplay


    def newCB(self):
        self.clearCB()
        self.settingsDialogRequestCB()

    def openFileCB(self):
        filePath   = QFileDialog.getOpenFileName(self, 
            "Open Morph", r".", "Morph Files (*.mor), Xml Files (*.xml)")
        if not filePath:
            return
        fp = Path(str(filePath)).abspath()
        self.centralW.read(str(fp))
        self.centralW.update()
        self.latestSaveFile = fp

    def saveFileCB(self):
        sf = self.latestSaveFile or r"."
        filePath   = QFileDialog.getSaveFileName(self, 
            "Save Morph", sf, "Morph Files (*.mor), Xml Files (*.xml)")
        if not filePath:
            return
        fp = Path(str(filePath)).abspath()
        self.centralW.write(fp)
        self.latestSaveFile = fp

    def resetViewCB(self):
        self.centralW.resetCamera()

    def makeToolBar(self):
        self.toolbar = QToolBar(self)
        self.toolbar.setFloatable(True)
        self.toolbar.setMovable(True)
        #self.toolbar.setAllowedAreas(Qt.TopDockWidgetArea)
        self.addToolBar(Qt.TopToolBarArea,self.toolbar)

        self.toolbar.addAction(E.getIcon("New"), 'New...', self.newCB)
        self.toolbar.addAction(E.getIcon("Folder"), 'Open...', self.openFileCB)
        self.toolbar.addAction(E.getIcon("save"), 'Save...', self.saveFileCB)
        self.toolbar.addSeparator()
        self.toolbar.addAction(E.getIcon("gear"), 'Settings...', self.settingsDialogRequestCB)
        self.toolbar.addSeparator()
        self.toolbar.addAction(E.getIcon("Picture-Save"), 'Export image...', self.exportImageCB)
        self.toolbar.addAction(E.getIcon("Film-Save"), 'Render...', self.renderDialogRequestCB)
        self.toolbar.addSeparator()
        self.toolbar.addAction(E.getIcon("Bomb"), 'Clear', self.clearCB)


    def exportImageCB(self):
        filePath   = QFileDialog.getSaveFileName(self, 
            "Export image", "", "Image Files (*.png *.jpg *.bmp)")
        if not filePath:
            return
        fp = Path(str(filePath)).abspath()
        im = self.centralW.renderFrame()
        im.save(str(fp))

    def settingsDialogRequestCB(self):
        a, b = self.getImages()
        self.settingsD.setup(a,b)
        self.settingsD.show()

    def renderDialogRequestCB(self):
        self.renderD.setup()
        self.renderD.show()

    def doRenderSequence(self,directory,filename,startFrame,endFrame,width,height,ext="png"):
        nFrames = endFrame - startFrame + 1
        tileW, tileH = 500, 500
        nTiles = (width//tileW) * (height//tileH)
        prog = QProgressDialog("Rendering sequence", "Abort", 1, nFrames*nTiles)
        prog.setWindowModality(Qt.WindowModal)
        prog.forceShow()
        saved = self.centralW.saveSettings()
        self.centralW.setSettings(dict(
                triangleDisplay = 0,
                vectorDisplay = 0,
                pointDisplay = 0,
            ))
        def cb(w,n,f):
            tileNum,totalTiles = n
            prog.setValue(f*nTiles+tileNum)
        for frame in range(startFrame,endFrame+1):
            frameCount = frame - startFrame
            if prog.wasCanceled(): break
            mix = float(frame-startFrame) / float(endFrame-startFrame)
            self.centralW.setMix(mix)
            im = self.centralW.renderFrame((width,height),callback=lambda w,n,f=frameCount: cb(w,n,f))
            fName = Path(directory) / ("%s.%05d.%s" % (filename,frame,ext))
            im.save(str(fName))
        prog.setValue(nFrames) 
        self.centralW.setSettings(saved)           

    def saveSnapshotCB(self):
        filePath   = QFileDialog.getSaveFileName(self, 
            "Save snapshot...", "", "Image Files (*.jpg, *.bmp, *.png, *.gif)")
        if not filePath:
            return
        savePixmap = self.centralW.renderPixmap(self.centralW.size().width(), 
                    self.centralW.size().height())
        savePixmap.save(filePath)

    def makeToolBoxDW(self):
        self.toolboxDW = QDockWidget("Toolbox")
        self.addDockWidget(Qt.LeftDockWidgetArea,self.toolboxDW)
        self.toolbox = QToolBox()
        self.toolboxDW.setWidget( self.toolbox )

        self.imageOptionsW = QWidget(self.toolbox)
        iLayout = QVBoxLayout()
        self.imageOptionsW.setLayout(iLayout)
        parameterLayout = QGridLayout()
        iLayout.addLayout(parameterLayout)
        self.toolbox.addItem(self.imageOptionsW,E.getIcon("gear"),"Parameters")

        if False:
            self.nameLE = QLineEdit("",self.toolbox)
            self.connect(self.nameLE,SIGNAL("textChanged (const QString&)"), self.nameCB)
            parameterLayout.addWidget(self.nameLE,0,1)
            parameterLayout.addWidget(QLabel("Name"),0,0)

        self.mixHS = QSlider(self)
        self.mixHS.setMinimum(0)
        self.mixHS.setMaximum(1000)
        self.mixHS.setTickInterval(100)
        ##self.mixHS.setGeometry(QRect(150,5,381,20))
        self.mixHS.setOrientation(Qt.Horizontal)
        self.mixHS.setSizePolicy(QSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed))
        self.mixHS.setMinimumHeight(20)
        self.connect(self.mixHS, SIGNAL("valueChanged(int)"), self.mixCB)
        parameterLayout.addWidget(self.mixHS,1,1)
        parameterLayout.addWidget(QLabel("Mix"),1,0)


        iLayout.addStretch()

    def initializeUi(self):
        #self.nameLE.setText(os.getenv("NAME","name"))
        pass

    def nameCB(self,s):
        pass
        #print "name", s

    def mixCB(self,v):
        self.centralW.setMix( clamp( float(v) / 1000.0, 0.0, 1.0 ) )

    def clearCB(self,*v):
        self.centralW.clear()

    def setMix(self,mix):
        self.mixHS.setValue( int(mix * 1000) )

    def getMix(self):
        return float( self.mixHS.value() ) / 1000.0

    def getImages(self):
        a = self.centralW.getImageA()
        b = self.centralW.getImageB()
        return (a,b)

    def setImages(self,a,b):
        self.centralW.loadImageA(a)
        self.centralW.loadImageB(b)
        self.centralW.update()

#-------------------------------------------------------------------------------


morphFragSrc1 = '''
uniform float     blend;
uniform sampler2D src;
uniform float     uGamma;

void main() {
    vec2 st           = gl_TexCoord[0].st;
    float u           = pow(st.s,uGamma);
    float v           = st.t;
    vec4 cA           = texture2D(src,  vec2(u,v));
    gl_FragColor      = vec4( cA.rgb, blend); //pow(blend,0.4));
}'''

morphVtxSrc2  = '''
uniform float vtxblend;
void main() {
    gl_Position = gl_ModelViewProjectionMatrix * 
        (gl_Vertex + vtxblend*vec4(gl_Normal,0.0));
    gl_TexCoord[0]  = gl_MultiTexCoord0;
}
'''

morphVtxSrc  = '''
uniform float vtxblend;
void main() {
    gl_Position  = ftransform(); 
    gl_Position += vtxblend * vec4(gl_Normal,0.0);
    gl_TexCoord[0]  = gl_MultiTexCoord0;
}
'''

morphFragSrc = '''
uniform float     txblend;
uniform sampler2D src;

void main() {
    vec2  st      = gl_TexCoord[0].st;
    vec4  cA      = texture2D(src, st);
    gl_FragColor  = vec4( cA.rgb, txblend); //pow(txblend,0.4));
}'''



#-------------------------------------------------------------------------------

class EditCursor(object):
    
    def __init__(self,canvas):
        self.canvas = canvas
        self.index = None
        self.endpoint = None
        self.inMiddle = False
    
    def __str__(self):
        if self.endpoint != None:
            es = ",endpoint='%s'" % self.endpoint
        else:
            es = ''
        if self.inMiddle:
            es = ",inMiddle=True"
        s = "%s(%s%s)" % (self.__class__.__name__,self.index,es)
        return s
    __repr__ = __str__

    def reset(self):
        self.index = self.last
    
    @property
    def last(self):
        self.index = len(self.canvas.samples)
        self._limit()
        return self.index
    
    def valid(self):
        return len(self.canvas.samples) > 4

    def _limit(self):
        n = len(self.canvas.samples)
        if n < 4:
            self.index = None
        if self.index > (n-1):
            self.index = (n-1)

    def goToLast(self):
        self.index = self.last

    def goToNext(self):
        self.index += 1
        self._limit()
        return self.index

    def goToPrev(self):
        self.index -= 1
        self._limit()
        return self.index

    @property
    def sample(self):
        if self.index != None and self.last != None and \
            self.index >= 4 and self.index <= self.last:
            return self.canvas.samples[self.index]
        else:
            return None
    
    def __int__(self):
        return self.index
    
    def set(self,value,endpoint=None,inMiddle=True):
        #print "SET",endpoint,inMiddle
        if isinstance(value,(int,float)):
            value = int(value)
        elif hasattr(value,"index"):
            self.index = value.index
        self._limit()
        self.endpoint = endpoint
        self.inMiddle = inMiddle

#-------------------------------------------------------------------------------


class MorphWidget(QGLShaderWidget):

    fontB18 = QFont("Helvetica", 18, QFont.Bold)

    def __init__(self, parent=None):
        super(MorphWidget, self).__init__(parent)
        self.setFocusPolicy(Qt.WheelFocus)
        
        self.objectA = None
        self.objectB = None
        self.imagePathA = None
        self.imagePathB = None

        self.camx, self.camy = -0.5, 0.5
        self.imageScale = 1.0
        self.imageAspect = 1.0
        self.uGamma = 1.0
        self.canvas = TriCanvas()

        # integer index into samples of "current sample"
        # Note that Samples 0 thru 3 are the fixed corners.
        self.cursor = EditCursor(self.canvas) 
        self.cursor.goToLast()
        
        self.clearColor = QColor(18,18,18)
        self.morphMix = 0.0
        
        self.drag = vec3(0)
        
        self.setWindowTitle("Morph")

        self.defineShader("morphA",   morphVtxSrc, morphFragSrc)
        self.defineShader("morphB",   morphVtxSrc, morphFragSrc)

        self.drawState = 'start'
        
        self.triangleDisplay  = 1
        self.vectorDisplay    = 1
        self.pointDisplay     = 1

        self.animationTimer = QTimer(self)
        self.connect(self.animationTimer, SIGNAL("timeout()"), self.animationCB)
        self.playback = None
        self.playbackParams = (None,)
        self.animationTimer.start(1000 // 30)

    def saveSettings(self):
        settings = dict()
        for a in "triangleDisplay vectorDisplay pointDisplay morphMix".split():
            settings[a] = getattr(self,a)
        return settings

    def setSettings(self,settings):
        for k,v in settings.items():
            setattr(self,k,v)

    def xml(self):
        
        root = Element("morph")
        for k,v in dict(version=FILEVERSION).items():
            root.attrib[k] = str(v).strip()
        
        xFileA = Element("fileA")
        xFileA.attrib['path']   = "%s" % self.imagePathA
        xFileA.attrib['aspect'] = "%s" % self.imageAspect
        xFileA.attrib['size']   = "%sx%s" % self.imageRes
        root.append(xFileA)

        xFileB = Element("fileB")
        xFileB.attrib['path']   = "%s" % self.imagePathB
        xFileB.attrib['aspect'] = "%s" % self.imageAspectB
        xFileB.attrib['size']   = "%sx%s" % self.imageResB
        root.append(xFileB)

        self.canvas.triangulate()

        xSamples = Element("samples")
        xSamples.attrib['num'] = "%s" % len(self.canvas.samples)
        root.append(xSamples)

        for num, sample in enumerate(self.canvas.samples):
            x = Element("sample")
            x.attrib['i']  = "%s" % num
            x.attrib['x']  = "%s" % sample.p.x
            x.attrib['y']  = "%s" % sample.p.y
            x.attrib['dx'] = "%s" % sample.n.x
            x.attrib['dy'] = "%s" % sample.n.y
            xSamples.append( x)

        xTriangles = Element("triangles")
        xTriangles.attrib['num'] = "%s" % len(self.canvas.triangulation.triangles)
        root.append(xTriangles)

        for tri in self.canvas.triangulation.triangles:
            xTri = Element("triangle")
            i0, i1, i2 = tri
            xTri.attrib['i0'] = "%s" % i0
            xTri.attrib['i1'] = "%s" % i1
            xTri.attrib['i2'] = "%s" % i2
            xTriangles.append(xTri)

        return root


    def quitCB(self):
        self.parent().close()

    def empty(self):
        return not self.cursor.valid()

    def doContextMenu(self,event,inSquare=False,alt=False):
        menu = QMenu(self)
        deleteEnable = (self.empty() == False)

        if inSquare:
            for i in [ 
                ("Delete", "Delete", deleteEnable), 
                ("Reverse", "Reverse", deleteEnable), 
                "---",
                ("Reset view", "Map", True),
                #("Quit","Quit"), 
                ]:
                if isinstance(i,tuple):
                    name, icon, en = i
                    subname = None
                    action = menu.addAction(E.getIcon(icon),name)
                    action.setEnabled(en)
                    self.connect(action, SIGNAL("triggered()"), 
                        lambda s=self,n=name: s.contextMenuCB(n))
                else:
                    if isinstance(i,str) and i.startswith("-"):
                        menu.addSeparator()
        
        if not inSquare or (inSquare and alt):
            if inSquare:
                menu.addSeparator()
                
            action = menu.addAction(E.getIcon("gear"),"Settings...")
            self.connect(action, SIGNAL("triggered()"), 
                lambda s=self: s.contextMenuCB("Settings"))
            
            action = menu.addAction(E.getIcon("save"),"Save")
            self.connect(action, SIGNAL("triggered()"), 
                lambda s=self: s.contextMenuCB("Save"))

            menu.addSeparator()

            action = menu.addAction(E.getIcon("Map"),"Reset view")
            self.connect(action, SIGNAL("triggered()"), 
                lambda s=self: s.contextMenuCB("Reset"))

           
        menu.exec_(event.globalPos())
        event.accept()

    def contextMenuCB(self,name,arg=None,param=None):

        if name.startswith("Delete"):
            self.deleteCurrentSample()
        elif name.startswith("Reverse"):
            self.reverseCurrentSample()
        elif name.startswith("Settings"):
            self.parent().settingsDialogRequestCB()
        elif name.startswith("Save"):
            self.parent().saveFileCB()
        elif name.startswith("Reset"):
            self.parent().resetViewCB()
    
    def deleteCurrentSample(self):
        self.canvas.delete(int(self.cursor))
        self.canvas.triangulate()
        self.update()

    def reverseCurrentSample(self):
        self.canvas.reverse(int(self.cursor))
        self.canvas.triangulate()
        self.update()

    def read(self,filename):
        root = parse(str(filename)).getroot()
        imageA = None
        imageB = None
        samples = list()
        triangles = list()
        
        for node in root:
            if node.tag in ( 'fileA', ):
                for k,v in node.items():
                    if k == 'path':
                        imageA = Path(v)
                #notes=node.text
            elif node.tag in ( 'fileB', ):
                for k,v in node.items():
                    if k == 'path':
                        imageB = Path(v)
            elif node.tag in ( 'samples', ):
                for xSamp in node:
                    if xSamp.tag != 'sample':
                        print "*** ignoring node '%s'" % xSamp.tag
                        continue
                    i = int(xSamp.attrib['i'])
                    x = float(xSamp.attrib['x'])
                    y = float(xSamp.attrib['y'])
                    dx = float(xSamp.attrib['dx'])
                    dy = float(xSamp.attrib['dy'])
                    s = TriSample(x,y,dx,dy)
                    #print i,s
                    samples.append( s )
                    
            elif node.tag in ( 'triangles', ):
                for xTri in node:
                    if xTri.tag != 'triangle':
                        print "*** ignoring node '%s'" % xTri.tag
                        continue
                    i0 = int(xTri.attrib['i0'])
                    i1 = int(xTri.attrib['i1'])
                    i2 = int(xTri.attrib['i2'])
                    triangles.append( (i0,i1,i2) )
            else:
                #raise Exception("unknown tag '%s'" % (node.tag,))
                print "*** ignoring unknown tag '%s'" % (node.tag,)

        if imageA and imageB and samples:
            self.loadImageA(imageA)
            self.loadImageB(imageB)
            self.canvas.clear()
            self.canvas.extend( samples[4:] )
            #for i,t in enumerate(self.canvas.samples):
            #    print i,t
            self.canvas.triangulate()
            self.cursor.reset()
            self.update()
            

    def write(self,filename):
        ElementTree(self.xml()).write(filename)

    def getImageA(self):
        return self.imagePathA

    def getImageB(self):
        return self.imagePathB

    def loadImageA(self,fileName):
        p = Path(fileName)
        res = getImageSize(p)
        if not res:
            raise Exception("bad image A '%s'" % p)
        self.imagePathA = p
        self.imageRes = res
        self.imageAspect = float(res[0]) / res[1] 
        self['morphA']['src'] = p
        self.objectA = self.canvas.makeGlObject()
    
    def loadImageB(self,fileName):
        p = Path(fileName)
        res = getImageSize(p)
        if not res:
            raise Exception("bad image B '%s'" % p)
        self.imagePathB = p
        self.imageResB = res
        self.imageAspectB = float(res[0]) / res[1] 
        self['morphB']['src'] = p
        #self.objectB = self.canvas.makeGlObject()

    def _readDefaultImages(self):
        p = E.appPath / 'images'
        self.loadImageA(p / 'testImageA.png')
        self.loadImageB(p / 'testImageB.png')


    def animationCB(self):
        #print ".",
        if self.playback == "morphMix": # This should be a list of playback object that have there own states
            incr = self.playbackParams
            self.morphMix += incr
            if self.morphMix >= 1.0 or self.morphMix <= 0.0:
                self.playback = None
                #print "done"
                if incr > 0.0:
                    self.morphMix = 1.0
                else:
                    self.morphMix = 0.0
            self.update()

    def clear(self):
        self.canvas.clear()
        self.cursor.reset()
        self.canvas.triangulate()
        self.update()

    def setMix(self,mix):
        self.morphMix = mix
        self.update()

    def minimumSizeHint(self):
        return QSize(150, 150)

    def sizeHint(self):
        return QSize(800, 800)

    def setClearColor(self, color):
        self.clearColor = color
        self.updateGL()

    def initializeGL(self):
        #super(MorphWidget, self).initializeGL()
        
        if not self.objectA:
            self._readDefaultImages()
        
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def screenToWorld(self,xs,ys):
        xw = ( 2.0 * (float(xs) / self.width()) - 1.0 ) \
                     / self.imageScale \
                         - (self.camx / self.imageScale)
        yw = (2.0 * (1.0 - (float(ys) / self.height())) - 1.0) \
                     / self.aspect / self.imageScale \
                         + (self.camy / self.imageScale)
        return (xw,yw)

    def paintGL(self):

        if self.canvas.triangulation == None:
            self.canvas.triangulate()

        self.qglClearColor(self.clearColor)
        glClear(GL_COLOR_BUFFER_BIT)

        glPushMatrix()
        glLoadIdentity()
        glTranslated(self.camx,-self.camy,0)
        glScale(self.imageScale, self.imageScale, 1.0)
        #print "scale",self.imageScale,self.imageAspect, self.camx, self.camy
  
        glLineWidth(1)
        glColor4f(0.1, 0.1, 0.1, 1.0)
        glBegin(GL_LINE_STRIP)
        glVertex3d(-10, 0, 0)
        glVertex3d( 10, 0, 0)
        glEnd()
        glBegin(GL_LINE_STRIP)
        glVertex3d(0, -10, 0)
        glVertex3d(0,  10, 0)
        glEnd()

        self['morphA']['vtxblend'] = 1.0 - self.morphMix
        self['morphA']['txblend'] = 1.0 # - self.morphMix
        self['morphB']['vtxblend'] = self.morphMix
        self['morphB']['txblend'] = self.morphMix
        
        self.applyShader("morphA")
        #glCallList(self.objectA)
       
        for triNum,t in enumerate(self.canvas.triangulation.triangles):
            glBegin(GL_TRIANGLES)
            for vtxNum,i in enumerate(t):
                x, y, _ = self.canvas.samples[i].p
                nx,ny,_ = self.canvas.samples[i].n
                glTexCoord2d(x, y)
                glVertex3d(x+self.morphMix*nx, y+self.morphMix*ny, 0)
            glEnd()

        self.removeShader()
        
        self.applyShader("morphB")
        #glCallList(self.objectA)
        for t in self.canvas.triangulation.triangles:
            glBegin(GL_TRIANGLES)
            for i in t:
                x, y, _ = self.canvas.samples[i].p
                nx,ny,_ = self.canvas.samples[i].n
                glTexCoord2d(x+nx, y+ny)
                glVertex3d(x+self.morphMix*nx, y+self.morphMix*ny, 0)
            glEnd()
        self.removeShader()
        
        if self.drawState in ( 'drag', 'press', ):
            if self.drawState == 'press':
                x0, y0 = self.screenToWorld(self.startPos.x(), self.startPos.y())
                x1, y1 = self.screenToWorld(self.currentPos.x(), self.currentPos.y())
                try:
                    xc, yc = clipToUnit(x0,y0,x1,y1)
                except:
                    #xc, yc = x1, y1
                    xc, yc = x0, y0
                if xc != x1 or yc != y1:
                    glLineWidth(3)
                    glColor4f(1.0, 1.0, 1.0, 0.2)
                    glBegin(GL_LINE_STRIP)
                    glVertex3d(xc, yc, 0)
                    glVertex3d(x1, y1, 0)
                    glEnd()
            elif self.drawState == 'drag':
                i = int(self.cursor)
                x, y, _ = self.canvas.samples[i].p
                nx,ny,_ = self.canvas.samples[i].n
                dragx, dragy = self.screenToWorld(self.dragPos.x(), self.dragPos.y())
                cx, cy = self.screenToWorld(self.currentPos.x(), self.currentPos.y())
                dx, dy = cx - dragx, cy - dragy
                if self.cursor.inMiddle:
                    x0, y0 = x + dx, y + dy
                    x1, y1 = x + nx + dx, y + ny + dy
                else:
                    if self.cursor.endpoint == 'p':
                        x0, y0 = x + dx, y + dy
                        x1, y1 = x + nx, y + ny
                    elif self.cursor.endpoint == 'n':
                        x0, y0 = x, y
                        x1, y1 = x + nx + dx, y + ny + dy
                xc, yc = x1, y1
                self.drag = vec3(dx,dy,0.0)
            if self.drawState == 'drag':
                glLineWidth(2)    
            else:
                glLineWidth(1)
            glColor4f(0.9, 0, 0.7, 1.0)
            glBegin(GL_LINE_STRIP)
            glVertex3d(x0, y0, 0)
            glVertex3d(xc, yc, 0)
            glEnd()

        if self.triangleDisplay:
            for tri in self.canvas.triangulation.triangles:
                i0, i1, i2 = tri
                s0, s1, s2 = [self.canvas.samples[i] for i in tri]
                pp0 = s0.p + self.morphMix * s0.n
                pp1 = s1.p + self.morphMix * s1.n
                pp2 = s2.p + self.morphMix * s2.n
                glLineWidth(1)
                glColor4f(0.0, 0.0, 0.0, 0.8)
                glBegin(GL_LINE_STRIP)
                glVertex3d(pp0.x, pp0.y, 0)
                glVertex3d(pp1.x, pp1.y, 0)
                glVertex3d(pp2.x, pp2.y, 0)
                glVertex3d(pp0.x, pp0.y, 0)
                glEnd()

        if self.vectorDisplay or self.pointDisplay:
            for i, sample in enumerate(self.canvas.samples):
                px, py = sample.p.x, sample.p.y
                dx, dy = sample.n.x, sample.n.y
                #print i,int(self.cursor),(i == int(self.cursor))
                if self.cursor == None:
                    highlight = 0
                else:
                    highlight = (i == int(self.cursor))
                clr = (1.0, 0.5, 0.7, 0.6)
                lw = 2
                if highlight:
                    clr = (1.0, 0.0, 0.7, 1.0)
                    lw = 3
                    if self.drawState == 'drag':
                        clr = (1.0, 0.0, 0.7, 0.5)
                        lw = 1
                if self.vectorDisplay:
                    glColor4f(*clr)
                    glLineWidth(lw)
                    glBegin(GL_LINE_STRIP)
                    glVertex3d(px,    py, 0)
                    glVertex3d(px+dx, py+dy, 0)
                    glEnd()
                if self.pointDisplay:
                    glColor4f(0.0, 0.0, 0.0, 0.5 + 0.5*highlight)
                    glPointSize(lw*2)
                    glBegin(GL_POINTS)
                    glVertex3d(px,    py, 0)
                    glEnd()
                    glColor4f(0.0, 0.0, 0.5, 0.5 + 0.5*highlight)
                    glPointSize(lw)
                    glBegin(GL_POINTS)
                    glVertex3d(px+dx, py+dy, 0)
                    glEnd()

        glPopMatrix()
        #self.canvas.dump()

    def renderFrame(self, finalSize=(3000,1500), tileSize=(500,500), callback=None):
        """
        """
        if finalSize[0] <= MAXRES[0] and finalSize[1] <= MAXRES[1]:
            return self.renderTile(finalSize[0], finalSize[1], 0.0, 1.0, 0.0, 1.0)
        
        canvas = Image.new("RGB",finalSize)
        nTilesU = finalSize[0] // tileSize[0]
        nTilesV = finalSize[1] // tileSize[1]
        totalTiles = nTilesU * nTilesV
        tileNum = 0
        for tileU in range(nTilesU):
            u0 = (tileU+0) * (1.0 / float(nTilesU))
            u1 = (tileU+1) * (1.0 / float(nTilesU))
            xo = int(u0 * finalSize[0])
            for tileV in range(nTilesV):
                v0 = (tileV+0) * (1.0 / float(nTilesV))
                v1 = (tileV+1) * (1.0 / float(nTilesV))
                yo = int( (1.0-v1) * finalSize[1])
                tileIm = self.renderTile(tileSize[0], tileSize[1], u0, u1, v0, v1)
                canvas.paste(tileIm,(xo,yo))
                if callback:
                    callback(self,(tileNum,totalTiles))
                tileNum += 1
        return canvas
       
    def renderTile(self, width, height, left, right, bottom, top):
        #print "RENDER: ",width, height, left, right, bottom, top

        if self.canvas.triangulation == None:
            self.canvas.triangulate()

        # Save viewport
        _,_,widthOrig,heightOrig = glGetIntegerv(GL_VIEWPORT)
        
        #resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(left, right, bottom, top, -1, 1)
        glMatrixMode(GL_MODELVIEW)

        self.qglClearColor(self.clearColor)
        glClear(GL_COLOR_BUFFER_BIT)

        glPushMatrix()
        glLoadIdentity()
  
        self['morphA']['vtxblend'] = 1.0 - self.morphMix
        self['morphA']['txblend'] = 1.0 
        self['morphB']['vtxblend'] = self.morphMix
        self['morphB']['txblend'] = self.morphMix
        
        self.applyShader("morphA")
        #glCallList(self.objectA)
        for triNum,t in enumerate(self.canvas.triangulation.triangles):
            glBegin(GL_TRIANGLES)
            for vtxNum,i in enumerate(t):
                x, y, _ = self.canvas.samples[i].p
                nx,ny,_ = self.canvas.samples[i].n
                glTexCoord2d(x, y)
                glVertex3d(x+self.morphMix*nx, y+self.morphMix*ny, 0)
            glEnd()
        self.removeShader()
        
        self.applyShader("morphB")
        #glCallList(self.objectA)
        for t in self.canvas.triangulation.triangles:
            glBegin(GL_TRIANGLES)
            for i in t:
                x, y, _ = self.canvas.samples[i].p
                nx,ny,_ = self.canvas.samples[i].n
                glTexCoord2d(x+nx, y+ny)
                glVertex3d(x+self.morphMix*nx, y+self.morphMix*ny, 0)
            glEnd()
        self.removeShader()

        if self.triangleDisplay:
            for tri in self.canvas.triangulation.triangles:
                i0, i1, i2 = tri
                s0, s1, s2 = [self.canvas.samples[i] for i in tri]
                pp0 = s0.p + self.morphMix * s0.n
                pp1 = s1.p + self.morphMix * s1.n
                pp2 = s2.p + self.morphMix * s2.n
                glLineWidth(1)
                glColor4f(0.0, 0.0, 0.0, 1.0)
                glBegin(GL_LINE_STRIP)
                glVertex3d(pp0.x, pp0.y, 0)
                glVertex3d(pp1.x, pp1.y, 0)
                glVertex3d(pp2.x, pp2.y, 0)
                glVertex3d(pp0.x, pp0.y, 0)
                glEnd()

        if self.vectorDisplay or self.pointDisplay:
            for i, sample in enumerate(self.canvas.samples):
                px, py = sample.p.x, sample.p.y
                dx, dy = sample.n.x, sample.n.y
                #print i,int(self.cursor),(i == int(self.cursor))
                if self.cursor == None:
                    highlight = 0
                else:
                    highlight = (i == int(self.cursor))
                clr = (1.0, 0.5, 0.7, 0.6)
                lw = 2
                if highlight:
                    clr = (1.0, 0.0, 0.7, 1.0)
                    lw = 3
                if self.vectorDisplay:
                    glColor4f(*clr)
                    glLineWidth(lw)
                    glBegin(GL_LINE_STRIP)
                    glVertex3d(px,    py, 0)
                    glVertex3d(px+dx, py+dy, 0)
                    glEnd()
                if self.pointDisplay:
                    glColor4f(0.0, 0.0, 0.0, 0.5 + 0.5*highlight)
                    glPointSize(lw*2)
                    glBegin(GL_POINTS)
                    glVertex3d(px,    py, 0)
                    glEnd()
                    glColor4f(0.0, 0.0, 0.5, 0.5 + 0.5*highlight)
                    glPointSize(lw)
                    glBegin(GL_POINTS)
                    glVertex3d(px+dx, py+dy, 0)
                    glEnd()


        glPopMatrix()
        
        self.resizeGL(widthOrig,heightOrig)

        pixels = glReadPixels(0,0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
        im = Image.new("RGBA",(width,height))
        im.fromstring(pixels)
        return im.transpose(Image.FLIP_TOP_BOTTOM)
        #.save(outfile)
        #print "...saved",Path(outfile).abspath()
        

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

        self.aspect = float(width) / height
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        if self.aspect <= 1.0:
            glOrtho(-self.aspect, self.aspect, -1, 1, -1, 1)
        else:
            glOrtho(-1, 1, -1.0 / self.aspect, 1.0 / self.aspect, -1, 1)

        #gluPerspective(60., width / float(height), 1, 1000.)

        glMatrixMode(GL_MODELVIEW)

    def keyPressEvent(self, event):
        #if event.modifiers() & Qt.AltModifier:
        if event.key() in (Qt.Key_T,):
            self.triangleDisplay = 1 - self.triangleDisplay
        elif event.key() in (Qt.Key_Home,):
            self.resetCamera()
        elif event.key() in (Qt.Key_V,):
            self.vectorDisplay = 1 - self.vectorDisplay
        elif event.key() in (Qt.Key_P,):
            self.pointDisplay = 1 - self.pointDisplay
        elif event.key() in ( Qt.Key_Left, Qt.Key_A, ): # mix=0
            self.parent().setMix(0.0)
        elif event.key() in ( Qt.Key_Right, Qt.Key_D, ): # mix=1
            self.parent().setMix(1.0)
        elif event.key() in ( Qt.Key_Down, Qt.Key_S, ): # mix=0.5
            self.parent().setMix(0.5)
        elif event.key() in ( Qt.Key_Up, Qt.Key_W, ): # mix = 1-mix
            self.parent().setMix(1.0 - self.parent().getMix())
        elif event.key() in ( Qt.Key_QuoteLeft, ): # animate mix toggle
            self.playback = "morphMix"
            if self.morphMix < 0.5:
                self.playbackParams = 0.15
            else:
                self.playbackParams = -0.15
        #print event.key()
        event.accept()
        self.update()

    def wheelEvent(self,event):
        oldScale = self.imageScale
        p = QPoint(event.pos())
        x, y = self.screenToWorld(p.x(), p.y())
        w = vec3(x,y,0)
        factor = 1.0 + 0.15 * (event.delta() // 120)
        self.imageScale = clamp(self.imageScale * factor,0.05,20.0)
        d = (self.imageScale - oldScale) * w
        self.camx -= d.x
        self.camy += d.y
        self.update()

    def resetCamera(self):
        self.camx, self.camy = -0.5, 0.5
        self.imageScale = 1.0
        self.update()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()
        self.currentPos = QPoint(event.pos())
        x, y = self.screenToWorld(self.currentPos.x(), self.currentPos.y())
        inSquare = (x>=0.0 and x<=1.0 and y>=0.0 and y<=1.0)
        if event.button() in (Qt.LeftButton,):
            if event.modifiers() & Qt.ControlModifier:
                if inSquare:
                    self.startPos = self.currentPos
                    self.drawState = 'press'
            elif event.modifiers() & Qt.AltModifier:
                pass
            elif event.modifiers() & Qt.ShiftModifier:
                pass
            else:
                self.doPick(x,y)
                self.update()

        elif event.button() in (Qt.MiddleButton,):
            if inSquare:
                self.doPick(x,y)
                self.dragPos = self.currentPos
                self.drawState = 'drag'
        
        elif event.button() in (Qt.RightButton,):
            if inSquare:
                self.doPick(x,y)
            self.doContextMenu(event,inSquare,event.modifiers() & Qt.AltModifier)

    def doPick(self,xWorld,yWorld):
        closestDist = None
        closestSample = None
        closestIndex = None
        closestEnd = None
        middle = False
        mouse = vec3(xWorld,yWorld,0.0)
        for i, sample in enumerate(self.canvas.samples):
            if i < 4: continue
            p0 = sample.p
            p1 = p0 + sample.n
            d0, d1 = (p0-mouse).length(),(p1-mouse).length()
            c = min(d0,d1)
            if closestDist == None or c < closestDist:
                closestDist = c
                closestSample = sample
                closestIndex = i
                closestEnd = ('p','n')[d1<d0]
                try:
                    ratio = d0 / (d0+d1)
                except:
                    ratio = 0.5
                middle = ratio > 0.333 and ratio < 0.666
        if closestIndex != None:
            self.cursor.set(closestSample,endpoint=closestEnd,inMiddle=middle)
            #print "pick",self.cursor,len(self.canvas.samples),closestDist,middle,closestEnd

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        self.currentPos = QPoint(event.pos())
        if event.buttons() & Qt.LeftButton:
            if event.modifiers() & Qt.AltModifier:
                self.camx += dx / 350.0
                self.camy += dy / 350.0
            elif event.modifiers() & Qt.ShiftModifier:
                self.morphMix += dx / 250.0
                self.morphMix = clamp(self.morphMix,0.0,1.0)
            else:
                pass
            self.updateGL()
        elif event.buttons() & Qt.MiddleButton:
            #x1, y1 = self.screenToWorld(self.currentPos.x(), self.currentPos.y())
            #x1, y1 = clipToUnit(x0,y0,x1,y1)
            #if event.modifiers() & Qt.ControlModifier:
            #    self.canvas.triangulate()
            self.updateGL()
        elif event.buttons() & Qt.RightButton:
            factor = 1.0 - 0.010 * dx
            s = self.imageScale * factor
            self.imageScale = clamp(s,0.05,20.0)
            self.updateGL()
        self.lastPos = event.pos()

    def mouseReleaseEvent(self, event):
        if self.drawState == 'press':
            x0, y0 = self.screenToWorld(self.startPos.x(), self.startPos.y())
            x1, y1 = self.screenToWorld(self.currentPos.x(), self.currentPos.y())
            x1, y1 = clipToUnit(x0,y0,x1,y1)
            sample = TriSample(x0, y0, x1-x0, y1-y0)
            self.canvas.insert( int(self.cursor) + 1, sample )
            self.cursor.goToNext()
            #for i,s in enumerate(self.canvas.samples):
            #    print i,s.p
            #print
            self.canvas.triangulate()
        elif self.drawState == 'drag':
            i = int(self.cursor)
            sample = self.canvas.samples[i]
            if self.cursor.inMiddle:
                middleOrEnd = "middle"
            else:
                middleOrEnd = self.cursor.endpoint
            sample.move(self.drag,which=middleOrEnd)
            self.canvas.triangulate()
        self.drawState = 'idle'
        self.update()

# ------------------------------------------------------------------------------
#


def gui():
    window = MainWindow()
    #deskRect = app.desktop().availableGeometry()
    #window.setGeometry(deskRect)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    gui()
