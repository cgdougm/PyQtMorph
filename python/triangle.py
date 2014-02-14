#!/usr/bin/env python

import sys

from path import path as Path

from cgkit.cgtypes import  vec3, vec4
import matplotlib.tri as Tri

from   OpenGL.GL         import *
from   OpenGL.GLU        import *

from   PyQt4.QtGui       import *
from   PyQt4.QtCore      import *
from   PyQt4.QtOpenGL    import *

import toolenv

if __name__=="__main__":
    app = QApplication(sys.argv)
    app.connect(app, SIGNAL("lastWindowClosed()"),
                app, SLOT("quit()"))
    app.env = toolenv.ToolEnv()

# ------------------------------------------------------------------------------

class TriSample:
    
    def __init__(self,x,y,nx,ny):
        self.p      = vec3(x,y,0)
        self.n      = vec3(nx,ny,0)
        self.index  = None

    def __str__(self):
        return "%s(%s,%s)" % (self.__class__.__name__,
            repr(self.p),repr(self.n))
    __repr__ = __str__

    def _getX(self):
        return self.p.x
    def _setX(self,x):
        self.p.x = x
    x = property(_getX,_setX,"X component")

    def _getY(self):
        return self.p.y
    def _setY(self,y):
        self.p.y = y
    y = property(_getY,_setY,"Y component")

    def _getPos(self):
        return self.p
    def _setPos(self,r,g=None,b=None):
        if g == None and b == None:
            if len(r) != 3:
                raise ValueError("expecting 3-tuple or vec3: %s" % r)
            x,y,z = r
        self.p = vec3(r,g,b)
    position = property(_getPos,_setPos,"XYZ position vector")

    def reverse(self):
        self.p      = self.p + self.n
        self.n      = -1.0 * self.n

    def move(self,v,which="middle"):
        if which == "n":
            self.n += v
        elif which == "p":
            self.p += v
            self.n -= v
        else:
            self.p += v

    def glVertex(self,blend=0.0):
        p = self.p + blend * self.n
        n = (1.0 - 2.0 * blend) * self.n
        glVertex3d(p.x, p.y, 0.0)
        glNormal3d(n.x, n.y, 0.0)
        glTexCoord2d(self.p.x, self.p.y)
        

# ------------------------------------------------------------------------------

class TriCanvas:
    
    def __init__(self,width=1.0,height=1.0):
        self.setSize(width,height)

    def __len__(self):
        return len(self.samples)

    def setSize(self,width,height):
        self.width  = width
        self.height = height
        self.clear()
    
    def clear(self):
        self.triangulation = None
        self.displayList = None
        self.samples = list() # array of TriSample objects
        # initial outside rectangle
        self.samples.extend( [
            TriSample(x, y, 0.0, 0.0) for x,y in 
                [ (0.0,1.0), (1.0,1.0), (1.0,0.0), (0.0,0.0),  ] 
        ] )
        self._renumber()

    def dump(self):
        print "======"
        print "Num points:",len(self.samples)
        for i,s in enumerate(self.samples):
            print i,s,"[%s]" % s.index
        print
        
    def addSample(self,x,y,nx,ny,u,v):
        sample = TriSample(x,y,nx,ny)
        self.samples.append( sample )
        self._renumber()
    
    def _renumber(self):
        for i, sample in enumerate(self.samples):
            sample.index = i

    def append(self,sample):
        self.samples.append( sample )
        self._renumber()
    
    def insert(self,i,sample):
        self.samples.insert( int(i), sample )
        self._renumber()
    
    def delete(self,i):
        if i < 4:
            raise Exception("can't delete corners (index=%d)" % i)
        if i > ( len(self.samples) - 1):
            raise Exception("index beyond end of list (index=%d, len=%d)" % (i,len(self.samples)))
        del self.samples[i]
        self._renumber()
    
    def reverse(self,i):
        if i < 4:
            raise Exception("can't delete corners (index=%d)" % i)
        if i > ( len(self.samples) - 1):
            raise Exception("index beyond end of list (index=%d, len=%d)" % (i,len(self.samples)))
        self.samples[i].reverse()
        self._renumber()
    
    def extend(self,samples):
        self.samples.extend( samples )
        self._renumber()
    
    def triangulate(self):
        xA = list(); yA = list()
        for sample in self.samples:
            xA.append(sample.x)
            yA.append(sample.y)
        self.triangulation = Tri.Triangulation(xA, yA)

    def callGlObject(self):
        if self.displayList == None:
            self.makeGlObject()
        glCallList(self.displayList)

    def makeGlObject(self):
        if self.displayList != None:
            glDeleteLists(self.displayList, 1)
        self.displayList = glGenLists(1)
        glNewList(self.displayList, GL_COMPILE)
        if self.triangulation == None:
            self.triangulate()
        for t in self.triangulation.triangles:
            glBegin(GL_TRIANGLES)
            for i in t:
                x, y, _ = self.samples[i].p
                nx,ny,_ = self.samples[i].n
                glNormal3d(nx, ny, 0)
                glTexCoord2d(x, y)
                glVertex3d(x*self.width, y*self.height, 0)
            glEnd()
        glEndList()
        return self.displayList

    def plot(self):
        import matplotlib.pyplot as Plt
        if self.triangulation == None:
            self.triangulate()
        Plt.figure()
        Plt.gca().set_aspect('equal')
        Plt.triplot(self.triangulation, 'bo-')
        Plt.title('Delaunay Triangulation')
        Plt.show()
    
# ------------------------------------------------------------------------------

def _test():
    import random
    u = random.uniform
    canvas = TriCanvas()
    for i in range(2000):
        canvas.addSample(u(0,1),u(0,1),u(0,1),u(0,1))
    canvas.plot()


# ------------------------------------------------------------------------------

class QGLTriangleCanvas(QWidget):

    defaultSize = QSize(640,480)

    def __init__(self, parent, plate=None, application=None):
        global app
        super(QGLTriangleCanvas,self).__init__(parent)
        self.setFocusPolicy(Qt.WheelFocus)
        if application:
            self.app = application
        else:
            self.app = app
        self.plate = None
        self.clear()
        if plate != None:
            self.loadPlate(plate)
        else:
            self.paintingSize = self.defaultSize 
        self.currentPos = QPoint(0,0)
        self.maxPixmapSize = QSize(640,480)
        
        self.canvas = TriCanvas(640,480)

        self.clipboard = self.app.clipboard()
        
        self.drawState = 'start'

    def sizeHint(self):
        return QSize(self.width,self.height)

    def _get_width(self):  return self.paintingSize.width()
    def _set_width(self,width):   self.paintingSize.setWidth(width)
    width = property(_get_width,_set_width)

    def _get_height(self): return self.paintingSize.height()
    def _set_height(self,height): self.paintingSize.setHeight(height)
    height = property(_get_height,_set_height)

    def clear(self):
        if self.plate:
            self.plate = QImage(self.plate.size(),QImage.Format_ARGB32)
        else:
            self.plate = QImage(self.defaultSize,QImage.Format_ARGB32)
        self.update()
    
    def quitCB(self):
        self.parent().close()

    def paintEvent(self, event):
        painter = QPainter(self)
        #painter.drawImage(QPoint(0,0), self.plate)
        if self.drawState == 'press':
            pen = QPen()
            pen.setColor(QColor(Qt.red))
            pen.setWidth(1)
            painter.setPen(pen)
            x0, y0 = self.startPos.x(), self.startPos.y()
            x1, y1 = self.currentPos.x(), self.currentPos.y()
            painter.drawLine (int(x0), int(y0), int(x1), int(y1))
        self.drawTriEdges(painter)
        
    def drawTriEdges(self,painter):
        penP = QPen()
        penP.setColor(QColor(Qt.black))
        penP.setWidth(2)
        pen0 = QPen()
        pen0.setWidth(16)
        #if not hasattr(self.canvas.triangulation,"triangles"):
        #    return
        if self.canvas.triangulation == None:
            self.canvas.triangulate()
        for tri in self.canvas.triangulation.triangles:
            i0, i1, i2 = tri
            s0, s1, s2 = [self.canvas.samples[i] for i in tri]
            poly = QPolygon()
            pt0 = QPoint(int(s0.x * self.width), int(s0.y * self.height))
            pt1 = QPoint(int(s1.x * self.width), int(s1.y * self.height))
            pt2 = QPoint(int(s2.x * self.width), int(s2.y * self.height))
            poly += pt0; poly += pt1; poly += pt2
            painter.setPen(penP)
            painter.drawPolygon(poly)
            
            c = QColor(int(255*s0.x),int(255*s0.y),0)
            #print i0,s0.p,c.name()
            pen0.setColor(c)
            painter.setPen(pen0)
            painter.drawPoint(QPoint(pt0))
        #print


    def loadPlate(self,imagePath):
        self.plate = QImage(imagePath)
        self.paintingSize = self.plate.size()
        self.resize(self.width,self.height)
    
    def keyPressEvent(self, event):
        #if event.modifiers() & Qt.AltModifier:
        event.accept()

    def mouseMoveEvent(self, event):
        self.currentPos = QPoint(event.pos())
        self.update()

    def mousePressEvent(self, event):
        self.currentPos = QPoint(event.pos())
        if event.button() in (Qt.LeftButton,):
            self.startPos = self.currentPos
            self.drawState = 'press'
            #if event.modifiers() & Qt.ControlModifier:
            #elif event.modifiers() & Qt.AltModifier:
            self.update()
        elif event.button() in (Qt.RightButton,):
            pass
        event.accept()

    def mouseReleaseEvent(self, event):
        if self.drawState == 'press':
            x0  = float(self.startPos.x())   / self.width
            y0  = float(self.startPos.y())   / self.height
            x1  = float(self.currentPos.x()) / self.width
            y1  = float(self.currentPos.y()) / self.height
            sample = TriSample(x0, y0, x1-x0, y1-y0)
            self.canvas.append( sample )
            self.canvas.triangulate()
        self.drawState = 'idle'
        self.update()
        self.canvas.dump()

    
class _Env(object):
    def getIcon(self,iconName):
        fileDir = Path(__file__).dirname()
        iconDir = fileDir / "images" / "icons"
        return QIcon("%s.%s.png" % (iconDir,iconName))
        

class MainWindow(QMainWindow):

    def __init__(self, plate=None, parent=None):
        super(MainWindow,self).__init__(parent)
        self.painting = QGLTriangleCanvas(self,plate)
        self.painting.E = _Env()
        self.setCentralWidget(self.painting)

    def genRandom(self):
        import random
        u = random.uniform        
        self.canvas = TriCanvas()

def main(args):
    resourceImageDir = Path(r'C:\Python26\Lib\site-packages')
    #resourceImageDir = Path(__file__).dirname()
    win=MainWindow(resourceImageDir / 'bricks.bmp')
    win.show()
    app.exec_()

# ------------------------------------------------------------------------------

if __name__=="__main__":
    main(sys.argv)