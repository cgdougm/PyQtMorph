
import sys
import os

from odict import OrderedDict

from path import path as Path

from PyQt4.QtOpenGL import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *

app = None
if __name__ == "__main__":
    app = QApplication(sys.argv)



# ------------------------------------------------------------------------------


class ToolEnv(object):
    """
    Global data
    Attributes:
        app         QApplication
        shows       list of Path objects to shows
        current     the directory that the app what started in
        platform    one of "windows", "linux" or "mac"
        clipboard   Clipboard object
        iconPath    path to all the icon PNGs
        appPath     path to the application
    Methods:
        getShots()  get a list of Path's to all shots for the given show
        getIcon()   return QIcon for the named icon
    """
    
    DATEFORMAT = "yyyyMMdd"

    def __init__(self,application=None):
        
        if application:
            self.app = application
        else:
            global app
            self.app = app
        
        self.current = Path().getcwd() # directory where the program was started
        
        self.platform = "linux"
        if sys.platform.lower().startswith("win"):
            self.platform = "windows"
        elif sys.platform.lower().startswith("dar"):
            self.platform = "mac"
        
        
        #myIcons = dict( [(path.namebase,QIcon(path)) for path in Path(r'C:\Documents and Settings\doug\My Documents\images\icons').files('*.png')])
        self.appPath = Path(__file__).dirname().abspath().dirname()
        if not self.appPath.isdir():
            self.appPath = Path().getcwd()
        self.iconPath = self.appPath / "images" / "icons"
        if not self.iconPath.isdir():
            raise Exception( "no icon directory '%s'" % self.iconPath)

    def getIcon(self,name):
        """
        Return a QIcon given a name of a PNG icon in the app's resources
        or blank icon if not found.
        """
        p = self.iconPath / ("%s.png" % name)
        if p.exists():
            icon = QIcon(str(p))
            icon.isDummy = False
        else:
            icon = QIcon()
            icon.isDummy = True
        return icon



def main():
    e = ToolEnv()
    print e.getAllShows()
    print e.appPath
    print e.getIcon("Palette")

if __name__ == '__main__':
    main()
