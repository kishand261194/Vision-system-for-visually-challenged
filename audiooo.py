from Foundation import *
import AppKit
import sys
from os import system
from hand import hand

class yu:
    t="HIi"
    n=0
    h1=0
    h2=0
    h3=0
    res=0
e=yu()

class SRDelegate(NSObject):
    def speechRecognizer_didRecognizeCommand_(self,sender,cmd):
        print "speechRecognizer_didRecognizeCommand_", cmd

        e.t=cmd
        sp(e.h1,e.h2,e.h3)



def sp(e1,e2,e3):
    print "hiii"
    try:
        e.h1=e1
        e.h2=e2
        e.h3=e3
    
        recog = AppKit.NSSpeechRecognizer.alloc().init()
        recog.setCommands_( [
                             u"coffee powder",
                             u"milk",
                             u"sugar",
                             u"Quit the test."])

        recog.setListensInForegroundOnly_(False)
        d = SRDelegate.alloc().init()
        recog.setDelegate_(d)
       
        if e.t=="coffee powder" :
            system("say  -v vicki it is in rack "+ str(e.h2))
            hand(1)
        
        
        elif e.t=="milk" :
            system("say  -v vicki it is in rack "+ str(e.h1))
            hand(2)
    
        
        elif e.t=="sugar":
            system("say  -v vicki it is in rack "+ str(e.h3))
            hand(3)
            
        else:
            system("say -v vicki What ingrident do you want coffee powder, milk , or sugar ")
            print "Listening..."
            recog.startListening()
            runLoop = NSRunLoop.currentRunLoop()
            runLoop.run()



    

# Now we need to enter the run loop...
    
    except ValueError:
         print "Try again..."