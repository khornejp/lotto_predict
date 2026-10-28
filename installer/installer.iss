; Inno Setup script: per-user install (no admin) of the PyInstaller onedir build.
; Compile with installer/build.ps1, which runs PyInstaller first.

#define AppName "LottoML"
#define AppVersion "0.1.0"

[Setup]
AppId={{6B0C2B7E-5F1D-4C3A-9E2B-7A4D1C9F0E31}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=LottoML-setup-{#AppVersion}-cpu
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\LottoML\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\LottoML.exe"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\LottoML.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\LottoML.exe"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

; User data (%LOCALAPPDATA%\LottoML: lotto.xlsx, checkpoints) is kept on uninstall on purpose.
