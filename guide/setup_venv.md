## 🐍 Python 3.8 Virtual Environment Setup (macOS & Ubuntu/WSL)

### 📦 macOS Instructions

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3.8**:

   ```bash
   brew install python@3.8
   brew link python@3.8 --force
   ```

3. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

4. **Create and activate the virtual environment**:

   ```bash
   python3.8 -m venv .venv
   source .venv/bin/activate
   ```

---

### 🐧 Ubuntu / WSL Instructions

> If Python 3.8 is not available on your system, follow these steps.

1. **Install prerequisites**:

   ```bash
   sudo apt update
   sudo apt install software-properties-common
   ```

2. **Add the deadsnakes PPA**:

   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   ```

3. **Install Python 3.8 and venv**:

   ```bash
   sudo apt install python3.8 python3.8-venv
   ```

5. **Create and activate the virtual environment**:

   ```bash
   python3.8 -m venv .venv
   source .venv/bin/activate
   ```
