# RapidEye — Windows Desktop (.EXE) Build & Packaging Guide

This document is the canonical step-by-step runbook for compiling and packaging the **RapidEye Security Monitor** into a standalone, one-click Windows Desktop Installer (`RapidEye_Setup_v1.0.0.exe`) on a **Windows 10 or Windows 11 machine with an NVIDIA GPU**.

---

## 1. Prerequisites (On the Windows Build Machine)

Make sure the Windows PC where you are building the `.exe` has the following installed:

| Tool | Minimum Version | Download Link | Purpose |
| :--- | :--- | :--- | :--- |
| **Python** | `3.11+` (or `3.12`/`3.13`) | [python.org](https://www.python.org/downloads/windows/) | Compiling backend & running PyInstaller |
| **Node.js** | `v20+` | [nodejs.org](https://nodejs.org/) | Compiling React static production build |
| **Inno Setup** | `v6.x` | [jrsoftware.org](https://jrsoftware.org/isdl.php) | Compiling the single one-click `.exe` setup wizard |
| **Git** | Any | [git-scm.com](https://git-scm.com/download/win) | Cloning the repository |

> ⚠️ **Important:** During Python installation, make sure to check **"Add python.exe to PATH"** so `python` and `pip` work directly in Command Prompt / PowerShell.

---

## 2. Step-by-Step Build Commands (PowerShell or Command Prompt)

Open **PowerShell** or **Command Prompt (`cmd`)** on Windows and run each step sequentially:

### Step 1: Clone or Copy the Repository onto Windows
If you haven't already transferred the repository from Linux to your Windows PC:
```cmd
git clone <your_repo_url>
cd rapid-eye-demo
```

---

### Step 2: Build the React Frontend (`web/dist/`)
Compile the React TypeScript dashboard into production static assets:
```cmd
cd web
npm install
npm run build
cd ..
```
* **Expected Output:** You should see `vite v8.x building for production... ✓ built in ~2s`. Verify that a `web\dist\` folder now exists with `index.html` inside it.

---

### Step 3: Create Python Virtual Environment & Install Dependencies
Create a clean Windows virtual environment and install our pinned requirements:
```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
> 💡 **Note on PyTorch & CUDA 12:** `requirements.txt` automatically pulls the CUDA-12 builds (`+cu126`) of PyTorch and `onnxruntime-gpu` along with NVIDIA runtime DLLs (`cublas`, `cudart`) directly via wheels. You do **not** need the full system CUDA Toolkit installed on Windows!

---

### Step 4: Test Run the Application Locally on Windows (Optional Check)
Before freezing into an `.exe`, verify that the desktop GUI (`pywebview`) and FastAPI static serving work on this Windows PC:
```cmd
python app_window.py
```
* **Expected Result:** A native Windows application window titled *"RapidEye Security Monitor (Local Desktop Edition)"* should open, rendering your React dashboard on `http://127.0.0.1:8001`.
* Close the window to shut down the test server cleanly (`Ctrl+C` if needed).

---

### Step 5: Freeze the Application with PyInstaller
Run PyInstaller using our pre-configured specification file (`rapid_eye_desktop.spec`):
```cmd
pyinstaller rapid_eye_desktop.spec --clean
```
* **Expected Time:** `~2 to 4 minutes` depending on disk speed.
* **What Happens:** PyInstaller collects Python, FastAPI, Uvicorn, PyTorch/ONNX Runtime DLLs, all `data/models/onnx/*.onnx` AI models, the **entire `assets/` directory of test videos**, `.env`, and `web/dist/` into a single standalone folder: `dist\RapidEye\`.
* **Verify Output:** Navigate inside `dist\RapidEye\` in Windows File Explorer and double-click `RapidEye.exe`. If the desktop window opens cleanly, your build is 100% successful!

---

### Step 6: Compile the One-Click Setup Wizard (`RapidEye_Setup_v1.0.0.exe`)
Now wrap the `dist\RapidEye\` folder into a single, professional Windows installer file using **Inno Setup Compiler**:

#### Option A: Via GUI (Easiest)
1. Double-click `setup_wizard.iss` in File Explorer (it opens inside Inno Setup Compiler).
2. Click the green **Compile (`Ctrl+F9`) ▶ button** in the top toolbar.

#### Option B: Via Command Prompt (Automated / CI/CD)
```cmd
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup_wizard.iss
```

* **Final Output:** Look inside the `Output\` folder created in your project root. You will find:
  🎉 **`RapidEye_Setup_v1.0.0.exe`** (`~300 MB to 400 MB`)

---

## 3. Where Are the Bundled Test Videos on the Client's Machine?

When the end-user installs `RapidEye_Setup_v1.0.0.exe`, all 4 test video files from `assets/` (`camera_1.mp4`, `camera_2.mp4`, `camera_4.mp4`, `test_video.mp4`) are installed right alongside the application into:
```
C:\Program Files\RapidEye\assets\
```
When the client clicks **Manage Cameras** $\rightarrow$ **Add Camera** inside the app, they can simply choose **Local File** and select any of the test `.mp4` videos located inside `C:\Program Files\RapidEye\assets\`, or type `assets/camera_1.mp4` under **URL**!

---

## 4. Client PC Prerequisites (For Brand-New Windows Installations)

If the client is installing `RapidEye_Setup_v1.0.0.exe` on a **brand-new, fresh out-of-the-box Windows 10 or Windows 11 installation**, they do **NOT** need to install Python, Node.js, or the 3 GB NVIDIA CUDA Toolkit (`nvcc`). All of those are bundled inside the `.exe`!

However, a brand-new Windows PC must have these **two standard OS prerequisites** installed:

| Prerequisite | Why It Is Required | Download / Check |
| :--- | :--- | :--- |
| **1. NVIDIA Display Driver** | Even if an RTX GPU is inside the PC, CUDA cannot communicate with it if Windows is using the generic *"Microsoft Basic Display Adapter"* driver. | Download and install the latest **NVIDIA GeForce / RTX Display Driver** from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx). Verify with `nvidia-smi` in PowerShell. |
| **2. Microsoft Visual C++ Redistributable (x64)** | Python 3.11+, PyTorch, and OpenCV (`cv2`) depend on `msvcp140.dll` and `vcruntime140.dll` from Microsoft. If missing on a blank OS, launching `RapidEye.exe` will say *"VCRUNTIME140.dll was not found."* | Download and install [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) (`~24 MB`) from Microsoft official servers. |

