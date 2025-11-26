#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.1),
    on November 08, 2025, at 22:09
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.1'
expName = 'hemo_10min'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\yiwei\\1_Pilot_MREye_Data\\1_anatomical-Protocol\\MREye_psychopy\\MREye_psychopy\\hemo\\hemo_10min.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "intro_waiting_trigger" ---
    Intro = visual.TextStim(win=win, name='Intro',
        text='In this session\n\nYou will slowly move your head towards\nUp 30 sec, Center 30 sec\nDown 30 sec, Center 30 sec\nLeft 30 sec, Center 30 sec\nRight 30 sec, Center 30 sec\nThen we repeat it again\n\nThe scan will last 10 minutes.',
        font='Arial',
        pos=(0, 0.0), draggable=True, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "show_instruction" ---
    Stable = visual.TextStim(win=win, name='Stable',
        text='Keep your head at the center',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    up = visual.TextStim(win=win, name='up',
        text='Slowly Move Up',
        font='Arial',
        pos=(0, 0.35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    center = visual.TextStim(win=win, name='center',
        text='Move back to the center',
        font='Arial',
        pos=(0, 0.35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    down = visual.TextStim(win=win, name='down',
        text='Slowly turn down',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    center_2 = visual.TextStim(win=win, name='center_2',
        text='Move back to the center',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    Left = visual.TextStim(win=win, name='Left',
        text='Slowly turn left',
        font='Arial',
        pos=(-0.35, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    center_3 = visual.TextStim(win=win, name='center_3',
        text='Move back to the center',
        font='Arial',
        pos=(-0.35, 0), draggable=True, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    right = visual.TextStim(win=win, name='right',
        text='Slowly turn right',
        font='Arial',
        pos=(0.35, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    center_4 = visual.TextStim(win=win, name='center_4',
        text='Move back to the center',
        font='Arial',
        pos=(0.35, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    
    # --- Initialize components for Routine "stable_end" ---
    stableEnd = visual.TextStim(win=win, name='stableEnd',
        text="Good job\nLet's have a rest\nKeep the head stable\nto the end of the scan",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "intro_waiting_trigger" ---
    # create an object to store info about Routine intro_waiting_trigger
    intro_waiting_trigger = data.Routine(
        name='intro_waiting_trigger',
        components=[Intro, key_resp],
    )
    intro_waiting_trigger.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for intro_waiting_trigger
    intro_waiting_trigger.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro_waiting_trigger.tStart = globalClock.getTime(format='float')
    intro_waiting_trigger.status = STARTED
    thisExp.addData('intro_waiting_trigger.started', intro_waiting_trigger.tStart)
    intro_waiting_trigger.maxDuration = None
    # keep track of which components have finished
    intro_waiting_triggerComponents = intro_waiting_trigger.components
    for thisComponent in intro_waiting_trigger.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro_waiting_trigger" ---
    intro_waiting_trigger.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Intro* updates
        
        # if Intro is starting this frame...
        if Intro.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Intro.frameNStart = frameN  # exact frame index
            Intro.tStart = t  # local t and not account for scr refresh
            Intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Intro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Intro.started')
            # update status
            Intro.status = STARTED
            Intro.setAutoDraw(True)
        
        # if Intro is active this frame...
        if Intro.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['s'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            intro_waiting_trigger.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_waiting_trigger.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_waiting_trigger" ---
    for thisComponent in intro_waiting_trigger.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro_waiting_trigger
    intro_waiting_trigger.tStop = globalClock.getTime(format='float')
    intro_waiting_trigger.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro_waiting_trigger.stopped', intro_waiting_trigger.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "intro_waiting_trigger" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    repeat = data.TrialHandler2(
        name='repeat',
        nReps=2.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(repeat)  # add the loop to the experiment
    thisRepeat = repeat.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRepeat.rgb)
    if thisRepeat != None:
        for paramName in thisRepeat:
            globals()[paramName] = thisRepeat[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRepeat in repeat:
        currentLoop = repeat
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRepeat.rgb)
        if thisRepeat != None:
            for paramName in thisRepeat:
                globals()[paramName] = thisRepeat[paramName]
        
        # --- Prepare to start Routine "show_instruction" ---
        # create an object to store info about Routine show_instruction
        show_instruction = data.Routine(
            name='show_instruction',
            components=[Stable, up, center, down, center_2, Left, center_3, right, center_4],
        )
        show_instruction.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for show_instruction
        show_instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        show_instruction.tStart = globalClock.getTime(format='float')
        show_instruction.status = STARTED
        thisExp.addData('show_instruction.started', show_instruction.tStart)
        show_instruction.maxDuration = None
        # keep track of which components have finished
        show_instructionComponents = show_instruction.components
        for thisComponent in show_instruction.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "show_instruction" ---
        # if trial has changed, end Routine now
        if isinstance(repeat, data.TrialHandler2) and thisRepeat.thisN != repeat.thisTrial.thisN:
            continueRoutine = False
        show_instruction.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 270.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Stable* updates
            
            # if Stable is starting this frame...
            if Stable.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Stable.frameNStart = frameN  # exact frame index
                Stable.tStart = t  # local t and not account for scr refresh
                Stable.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Stable, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Stable.started')
                # update status
                Stable.status = STARTED
                Stable.setAutoDraw(True)
            
            # if Stable is active this frame...
            if Stable.status == STARTED:
                # update params
                pass
            
            # if Stable is stopping this frame...
            if Stable.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Stable.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    Stable.tStop = t  # not accounting for scr refresh
                    Stable.tStopRefresh = tThisFlipGlobal  # on global time
                    Stable.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Stable.stopped')
                    # update status
                    Stable.status = FINISHED
                    Stable.setAutoDraw(False)
            
            # *up* updates
            
            # if up is starting this frame...
            if up.status == NOT_STARTED and tThisFlip >= 30-frameTolerance:
                # keep track of start time/frame for later
                up.frameNStart = frameN  # exact frame index
                up.tStart = t  # local t and not account for scr refresh
                up.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(up, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'up.started')
                # update status
                up.status = STARTED
                up.setAutoDraw(True)
            
            # if up is active this frame...
            if up.status == STARTED:
                # update params
                pass
            
            # if up is stopping this frame...
            if up.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > up.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    up.tStop = t  # not accounting for scr refresh
                    up.tStopRefresh = tThisFlipGlobal  # on global time
                    up.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'up.stopped')
                    # update status
                    up.status = FINISHED
                    up.setAutoDraw(False)
            
            # *center* updates
            
            # if center is starting this frame...
            if center.status == NOT_STARTED and tThisFlip >= 60-frameTolerance:
                # keep track of start time/frame for later
                center.frameNStart = frameN  # exact frame index
                center.tStart = t  # local t and not account for scr refresh
                center.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(center, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'center.started')
                # update status
                center.status = STARTED
                center.setAutoDraw(True)
            
            # if center is active this frame...
            if center.status == STARTED:
                # update params
                pass
            
            # if center is stopping this frame...
            if center.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > center.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    center.tStop = t  # not accounting for scr refresh
                    center.tStopRefresh = tThisFlipGlobal  # on global time
                    center.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'center.stopped')
                    # update status
                    center.status = FINISHED
                    center.setAutoDraw(False)
            
            # *down* updates
            
            # if down is starting this frame...
            if down.status == NOT_STARTED and tThisFlip >= 90-frameTolerance:
                # keep track of start time/frame for later
                down.frameNStart = frameN  # exact frame index
                down.tStart = t  # local t and not account for scr refresh
                down.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(down, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'down.started')
                # update status
                down.status = STARTED
                down.setAutoDraw(True)
            
            # if down is active this frame...
            if down.status == STARTED:
                # update params
                pass
            
            # if down is stopping this frame...
            if down.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > down.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    down.tStop = t  # not accounting for scr refresh
                    down.tStopRefresh = tThisFlipGlobal  # on global time
                    down.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'down.stopped')
                    # update status
                    down.status = FINISHED
                    down.setAutoDraw(False)
            
            # *center_2* updates
            
            # if center_2 is starting this frame...
            if center_2.status == NOT_STARTED and tThisFlip >= 120-frameTolerance:
                # keep track of start time/frame for later
                center_2.frameNStart = frameN  # exact frame index
                center_2.tStart = t  # local t and not account for scr refresh
                center_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(center_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'center_2.started')
                # update status
                center_2.status = STARTED
                center_2.setAutoDraw(True)
            
            # if center_2 is active this frame...
            if center_2.status == STARTED:
                # update params
                pass
            
            # if center_2 is stopping this frame...
            if center_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > center_2.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    center_2.tStop = t  # not accounting for scr refresh
                    center_2.tStopRefresh = tThisFlipGlobal  # on global time
                    center_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'center_2.stopped')
                    # update status
                    center_2.status = FINISHED
                    center_2.setAutoDraw(False)
            
            # *Left* updates
            
            # if Left is starting this frame...
            if Left.status == NOT_STARTED and tThisFlip >= 150-frameTolerance:
                # keep track of start time/frame for later
                Left.frameNStart = frameN  # exact frame index
                Left.tStart = t  # local t and not account for scr refresh
                Left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Left.started')
                # update status
                Left.status = STARTED
                Left.setAutoDraw(True)
            
            # if Left is active this frame...
            if Left.status == STARTED:
                # update params
                pass
            
            # if Left is stopping this frame...
            if Left.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Left.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    Left.tStop = t  # not accounting for scr refresh
                    Left.tStopRefresh = tThisFlipGlobal  # on global time
                    Left.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Left.stopped')
                    # update status
                    Left.status = FINISHED
                    Left.setAutoDraw(False)
            
            # *center_3* updates
            
            # if center_3 is starting this frame...
            if center_3.status == NOT_STARTED and tThisFlip >= 180-frameTolerance:
                # keep track of start time/frame for later
                center_3.frameNStart = frameN  # exact frame index
                center_3.tStart = t  # local t and not account for scr refresh
                center_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(center_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'center_3.started')
                # update status
                center_3.status = STARTED
                center_3.setAutoDraw(True)
            
            # if center_3 is active this frame...
            if center_3.status == STARTED:
                # update params
                pass
            
            # if center_3 is stopping this frame...
            if center_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > center_3.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    center_3.tStop = t  # not accounting for scr refresh
                    center_3.tStopRefresh = tThisFlipGlobal  # on global time
                    center_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'center_3.stopped')
                    # update status
                    center_3.status = FINISHED
                    center_3.setAutoDraw(False)
            
            # *right* updates
            
            # if right is starting this frame...
            if right.status == NOT_STARTED and tThisFlip >= 210-frameTolerance:
                # keep track of start time/frame for later
                right.frameNStart = frameN  # exact frame index
                right.tStart = t  # local t and not account for scr refresh
                right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right.started')
                # update status
                right.status = STARTED
                right.setAutoDraw(True)
            
            # if right is active this frame...
            if right.status == STARTED:
                # update params
                pass
            
            # if right is stopping this frame...
            if right.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > right.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    right.tStop = t  # not accounting for scr refresh
                    right.tStopRefresh = tThisFlipGlobal  # on global time
                    right.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'right.stopped')
                    # update status
                    right.status = FINISHED
                    right.setAutoDraw(False)
            
            # *center_4* updates
            
            # if center_4 is starting this frame...
            if center_4.status == NOT_STARTED and tThisFlip >= 240-frameTolerance:
                # keep track of start time/frame for later
                center_4.frameNStart = frameN  # exact frame index
                center_4.tStart = t  # local t and not account for scr refresh
                center_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(center_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'center_4.started')
                # update status
                center_4.status = STARTED
                center_4.setAutoDraw(True)
            
            # if center_4 is active this frame...
            if center_4.status == STARTED:
                # update params
                pass
            
            # if center_4 is stopping this frame...
            if center_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > center_4.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    center_4.tStop = t  # not accounting for scr refresh
                    center_4.tStopRefresh = tThisFlipGlobal  # on global time
                    center_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'center_4.stopped')
                    # update status
                    center_4.status = FINISHED
                    center_4.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                show_instruction.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in show_instruction.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "show_instruction" ---
        for thisComponent in show_instruction.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for show_instruction
        show_instruction.tStop = globalClock.getTime(format='float')
        show_instruction.tStopRefresh = tThisFlipGlobal
        thisExp.addData('show_instruction.stopped', show_instruction.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if show_instruction.maxDurationReached:
            routineTimer.addTime(-show_instruction.maxDuration)
        elif show_instruction.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-270.000000)
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'repeat'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "stable_end" ---
    # create an object to store info about Routine stable_end
    stable_end = data.Routine(
        name='stable_end',
        components=[stableEnd],
    )
    stable_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for stable_end
    stable_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    stable_end.tStart = globalClock.getTime(format='float')
    stable_end.status = STARTED
    thisExp.addData('stable_end.started', stable_end.tStart)
    stable_end.maxDuration = None
    # keep track of which components have finished
    stable_endComponents = stable_end.components
    for thisComponent in stable_end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "stable_end" ---
    stable_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 70.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *stableEnd* updates
        
        # if stableEnd is starting this frame...
        if stableEnd.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            stableEnd.frameNStart = frameN  # exact frame index
            stableEnd.tStart = t  # local t and not account for scr refresh
            stableEnd.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stableEnd, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'stableEnd.started')
            # update status
            stableEnd.status = STARTED
            stableEnd.setAutoDraw(True)
        
        # if stableEnd is active this frame...
        if stableEnd.status == STARTED:
            # update params
            pass
        
        # if stableEnd is stopping this frame...
        if stableEnd.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > stableEnd.tStartRefresh + 70-frameTolerance:
                # keep track of stop time/frame for later
                stableEnd.tStop = t  # not accounting for scr refresh
                stableEnd.tStopRefresh = tThisFlipGlobal  # on global time
                stableEnd.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stableEnd.stopped')
                # update status
                stableEnd.status = FINISHED
                stableEnd.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            stable_end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in stable_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "stable_end" ---
    for thisComponent in stable_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for stable_end
    stable_end.tStop = globalClock.getTime(format='float')
    stable_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('stable_end.stopped', stable_end.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if stable_end.maxDurationReached:
        routineTimer.addTime(-stable_end.maxDuration)
    elif stable_end.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-70.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
