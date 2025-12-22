# Music Genres Classification - Compatibility

## Compatibility Issues
- The **Music Genres Classification (C++)** is designed to run on **Linux**.
- **On Windows, application and tests must be run inside Windows Subsystem for Linux (WSL).**
- **Linux version must be no less than Ubunut 24.04 or any other equivalent distribution (Debian/Fedora/Arch).**

### Fully Supported Platforms
- **Linux (Ubuntu/Debian)**
- **Windows (via WSL)**

### Not Supported
- **Native Windows (MSVC, MinGW)**
  - Aubio (`aubio`) requires manual setup on naitve Windows.
  - Google Test (`gtest`) requires manual setup on naitve Windows.
  - Profiling tools like `valgrind` are not available on Windows.
- **macOS**
  - `cuda` is not available on macOs.

## Installing WSL with Ubuntu
Open PowerShell as Administrator and run
```powershell
wsl --install -d Ubuntu
```
For Windows 10 users, ensure you have **WSL 2** installed:
```powershell
wsl --set-default-version 2
```

## Next Steps
For details on testing and validation methods used in this system, see [Testing & Validation](TESTING.md).

For instructions on building and running the project, see the [Building Guide](BUILD.md).
