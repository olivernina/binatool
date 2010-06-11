#!/usr/bin/env python

# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * Neither the name of cvtypes's copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# ----------------------------------------------------------------------------
""" Binatool is a binarization tool to help you binarize any kind of images through different algorithms
"""

DOCLINES = __doc__.split("\n")


from distutils.core import setup
import py2exe


#import modulefinder
#modulefinder.AddPackagePath("toolbar", "test")



CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: End Users/Desktop
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Natural Language :: English
Operating System :: OS Independent
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Multimedia :: Graphics
"""

#import pygtk
#pygtk.require('2.0')
#import gtk
#import os, sys, getopt, ConfigParser, string, gc
#import random, urllib, gobject, gettext, locale
#import stat, time, subprocess, shutil, filecmp
#import tempfile, socket, md5, threading




includes = ["pygtk","os","sys","getopt","ConfigParser", "string", "gc","random", "urllib", "gobject", "gettext", "locale",
            "stat", "time", "subprocess", "shutil", "filecmp","tempfile", "socket", "md5", "threading","cairo","gtk",
            "pango", "pangocairo", "atk", "gobject",
            "numpy","cvtypes","cvtools","ctypes","toolbar"] #"gtk._gtk","pango", "pangocairo", "atk", "gobject"

excludes = ["pywin", "pywin.debugger", "pywin.debugger.dbgcon",
            "pywin.dialogs", "pywin.dialogs.list",
            "Tkconstants","Tkinter","tcl",
           
             ]
packages = []
 
setup(name = 'binatool',
    version = '0.1.0',
    description = DOCLINES[0],
    author = 'Multiple Authors',
    url = 'http://code.google.com/p/binatool/',
    license = 'MIT License',
    platforms = 'OS Independent, Windows, Linux, MacOS',
    classifiers = filter(None, CLASSIFIERS.split('\n')),
    long_description = "\n".join(DOCLINES[2:]),
#    data_files=[('../main/images', ['README'])],
    windows=[
             {
              'script':'../main/binatool.py',
              'icon_resources':[(1,"bina.ico")],
             }
            ], 
    options = {"py2exe": {"compressed": 1,
                          "optimize": 0,
                          "ascii": 1,
                          "bundle_files": 1,
                          "packages": 'encodings',
                          "includes": includes,
                          "excludes": excludes
                          }},
)




#setup(
#    name = 'handytool',
#    description = 'Some handy tool',
#    version = '1.0',
#
#    windows = [
#                  {
#                      'script': 'handytool.py',
#                      'icon_resources': [(1, "handytool.ico")],
#                  }
#              ],
#
#    options = {
#                  'py2exe': {
#                      'packages':'encodings',
#                      'includes': 'cairo, pango, pangocairo, atk, gobject',
#                  }
#              },
#
#    data_files=[
#                   'handytool.glade',
#                   'readme.txt'
#               ]
#)

  
