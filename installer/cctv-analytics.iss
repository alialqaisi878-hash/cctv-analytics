; Inno Setup script for the CCTV Analytics Windows installer.
; Build:  ISCC.exe /DAppVersion=1.0.0 installer\cctv-analytics.iss   (output: dist\CCTV-Analytics-Setup.exe)
; GitHub Actions builds this automatically (.github/workflows/windows-installer.yml).

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppName "CCTV Analytics"

[Setup]
AppId={{6D3F2B71-4C8E-4E0B-9B57-1F2A8C3D9E41}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=CCTV Analytics
DefaultDirName={localappdata}\CCTVAnalytics
DisableDirPage=auto
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=CCTV-Analytics-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#AppName}
UninstallDisplayIcon={sys}\shell32.dll,22

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "autostart"; Description: "Start automatically when Windows starts"; GroupDescription: "Startup:"; Flags: unchecked

[Files]
Source: "..\app\*"; DestDir: "{app}\app"; Excludes: "__pycache__,*.pyc"; Flags: recursesubdirs ignoreversion
Source: "..\run.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\config.example.yaml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\installer\setup.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "..\installer\launch.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\samples\*.mp4"; DestDir: "{app}\samples"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\models\*.pt"; DestDir: "{app}\models"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{autoprograms}\{#AppName}\{#AppName}"; Filename: "{app}\launch.bat"; WorkingDir: "{app}"; IconFilename: "{sys}\shell32.dll"; IconIndex: 22
Name: "{autoprograms}\{#AppName}\Settings (config.yaml)"; Filename: "{sys}\notepad.exe"; Parameters: """{app}\config.yaml"""
Name: "{autoprograms}\{#AppName}\Repair {#AppName}"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\setup.ps1"" -AppDir ""{app}"""
Name: "{autoprograms}\{#AppName}\Install folder"; Filename: "{app}"
Name: "{autoprograms}\{#AppName}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\launch.bat"; WorkingDir: "{app}"; IconFilename: "{sys}\shell32.dll"; IconIndex: 22; Tasks: desktopicon
Name: "{userstartup}\{#AppName}"; Filename: "{app}\launch.bat"; Parameters: "--background"; WorkingDir: "{app}"; Tasks: autostart

[Run]
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\setup.ps1"" -AppDir ""{app}"""; \
  StatusMsg: "Installing Python and AI libraries (5-15 minutes, needs internet)..."; Flags: waituntilterminated
Filename: "{app}\launch.bat"; Description: "Start {#AppName} now"; WorkingDir: "{app}"; Flags: postinstall nowait skipifsilent shellexec

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\app"
Type: files; Name: "{app}\install.log"
Type: files; Name: "{app}\setup.lock"
; config.yaml and data\ (your statistics) are kept on uninstall; delete the folder manually to remove them.
