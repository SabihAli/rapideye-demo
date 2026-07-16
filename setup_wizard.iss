; Inno Setup Script Template (`setup_wizard.iss`)
; Compiles the frozen PyInstaller output (`dist/RapidEye/`) into a single
; standalone Windows setup wizard (`RapidEye_Setup_v1.0.0.exe`).

[Setup]
AppName=RapidEye Security Monitor
AppVersion=1.0.0
AppPublisher=RapidEye AI Systems
DefaultDirName={autopf}\RapidEye
DefaultGroupName=RapidEye
DisableProgramGroupPage=yes
OutputBaseFilename=RapidEye_Setup_v1.0.0
SetupIconFile=rapideye.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Source takes all files inside PyInstaller output folder (`dist/RapidEye/*`),
; including `assets/*` test videos, AI models, and frontend UI, copying them to `{app}` (`C:\Program Files\RapidEye\`)
Source: "dist\RapidEye\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\RapidEye Security Monitor"; Filename: "{app}\RapidEye.exe"
Name: "{group}\Uninstall RapidEye"; Filename: "{uninstallexe}"
Name: "{autodesktop}\RapidEye Security Monitor"; Filename: "{app}\RapidEye.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\RapidEye.exe"; Description: "{cm:LaunchProgram,RapidEye Security Monitor}"; Flags: nowait postinstall skipifsilent
