============================================================
VisionGaurdAI
AI-Powered Security Monitoring System
=====================================

Version: 1.0

Developers:
Shivansh Agrawal (26CA057)
Kushagra Singh (26CA065)

Program:
13th Gurugram Police Cyber Security Summer Internship Program
(GPCSSI 2026)

============================================================
PROJECT OVERVIEW
================

VisionGaurdAI is an Artificial Intelligence and Computer Vision
based security monitoring system designed to provide real-time
surveillance and threat awareness.

The system continuously monitors a camera feed, detects faces,
identifies known and unknown individuals, detects objects,
highlights potentially hazardous items, and provides a visual
dashboard for security monitoring.

The project combines:

• Computer Vision
• Artificial Intelligence
• Face Detection
• Face Recognition
• Object Detection
• Threat Assessment
• Event Logging
• Security Monitoring

============================================================
FEATURES
========

✓ Live Camera Monitoring

✓ Face Detection

✓ Face Recognition

✓ Unknown Person Detection

✓ Object Detection

✓ Hazardous Object Identification

✓ Threat Assessment

✓ Event Logging

✓ SQLite Database Integration

✓ Modern Dashboard Interface

✓ Real-Time Status Monitoring

============================================================
SYSTEM REQUIREMENTS
===================

Operating System:
• Windows 10 / Windows 11

Python Version:
• Python 3.10 or newer

Required Libraries:

• opencv-python
• ultralytics
• numpy
• Pillow

============================================================
INSTALLATION
============

Step 1:
Open Command Prompt.

Step 2:
Navigate to the project directory.

Example:

cd "C:\Users\HP\Desktop\Shivansh Packer\SentinelVisionAI"

Step 3:
Install all required dependencies.

pip install -r requirements.txt

Wait until installation completes.

============================================================
PROJECT STRUCTURE
=================

VisionGaurdAI/

│
├── main.py
├── config.py
├── requirements.txt
├── README.txt
│
├── data/
│   ├── security.db
│   ├── faces/
│   ├── models/
│   └── screenshots/
│
├── modules/
│   ├── camera.py
│   ├── database.py
│   ├── face_recognition_module.py
│   ├── object_detection.py
│   ├── threat_assessment.py
│   └── alert_system.py
│
├── ui/
│   └── main_window.py
│
└── face_register.py

============================================================
IMPORTANT FILES
===============

## main.py

Main application file.
Starts the complete VisionGaurdAI system.

## config.py

Contains all project settings.

## register_face.py

Used to register new persons.

## security.db

Stores registered persons and logs.

## main_window.py

Graphical User Interface (GUI).

============================================================
STARTING THE APPLICATION
========================

Method 1 (Recommended)

Open Command Prompt.

Navigate to the project folder.

Example:

cd "C:\Users\HP\Desktop\Shivansh Packer\SentinelVisionAI"

Run:

python main.py

The following will open:

✓ AI Security Dashboard

✓ Live Camera Feed

✓ Face Detection

✓ Object Detection

✓ Threat Monitoring

✓ System Statistics

============================================================
ALTERNATIVE METHOD
==================

You may also run:

main.py

by double-clicking it.

However, Command Prompt is recommended because
error messages remain visible.

============================================================
REGISTERING A NEW PERSON
========================

To add a person to the database:

Open Command Prompt.

Navigate to the project folder.

Run:

python register_face.py

Steps:

1. Enter person's name.
2. Click Browse.
3. Select image.
4. Click Register.

The system will:

✓ Save the image in data/faces/

✓ Add the person to the database

✓ Make the person available for recognition

============================================================
DATABASE INFORMATION
====================

Database Type:
SQLite

Database File:
data/security.db

Stores:

• Registered Persons

• Face Information

• Detection Logs

• Alert History

============================================================
TROUBLESHOOTING
===============

Problem:
Application closes immediately.

Solution:

Run from Command Prompt:

python main.py

The error message will remain visible.

---

Problem:
Camera not detected.

Solution:

1. Check webcam connection.
2. Verify camera permissions.
3. Check CAMERA_INDEX in config.py.

---

Problem:
Face not recognized.

Solution:

1. Use a clear image.
2. Ensure proper lighting.
3. Register image again.
4. Restart application.

---

Problem:
Python command not working.

Try:

py main.py

or

py register_face.py

If still not working:

Reinstall Python and enable:

"Add Python to PATH"

during installation.

============================================================
OUTPUTS
=======

The system provides:

• Live camera feed

• Face detection boxes

• Face recognition labels

• Object detection labels

• Threat monitoring information

• Event statistics

• Security dashboard

============================================================
PROJECT INFORMATION
===================

Project Name:
VisionGaurdAI

Developer:
Shivansh Agrawal

Roll Number:
26CA057

Domain:
Artificial Intelligence
Computer Vision
Cyber Security

Program:
13th Gurugram Police Cyber Security Summer Internship Program
(GPCSSI 2026)

============================================================
END OF FILE
===========
