#!/bin/sh
printf "\033]0;Installer\007"
clear
rm *.bat

# Function to create or activate a virtual environment
prepare_install() {
    if [ -d ".venv" ]; then
        echo "Venv found. This implies Applio has been already installed or this is a broken install"
        printf "Do you want to execute run-applio.sh? (Y/N): " >&2
        read -r r
        r=$(echo "$r" | tr '[:upper:]' '[:lower:]')
        if [ "$r" = "y" ]; then
            chmod +x run-applio.sh  
            ./run-applio.sh && exit 1
        else
            echo "Ok! The installation will continue. Good luck!"
        fi
        if [ -f ".venv/bin/activate" ]; then
            . .venv/bin/activate  
        else
            echo "Venv exists but activation file not found, re-creating venv..."
            rm -rf .venv
            create_venv  
        fi
    else
        create_venv
    fi
}

# Function to create the virtual environment and install dependencies
create_venv() {
    echo "Creating venv..."
    requirements_file="requirements.txt"
    echo "Checking if python exists"
    if command -v python3.10 > /dev/null 2>&1; then
        py=$(which python3.10)
        echo "Using python3.10"
    else
        if python --version | grep -qE "3\.(7|8|9|10)\."; then
            py=$(which python)
            echo "Using python"
        else
            echo "Please install Python3 or 3.10 manually."
            exit 1
        fi
    fi
    # Try to create the env
    $py -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Error creating the virtual environment. Check Python installation or permissions."
        exit 1
    fi
    . .venv/bin/activate
    if [ $? -ne 0 ]; then
        echo "Error activating the virtual environment. Check if it was created properly."
        exit 1
    fi

    # Installs pip using ensurepip or get-pip.py
    echo "Installing pip..."
    python -m ensurepip --upgrade
    if [ $? -ne 0 ]; then
        echo "Error with ensurepip, attempting manual pip installation..."
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python get-pip.py
        if [ $? -ne 0 ];then
            echo "Failed to install pip manually. Check permissions and internet connection."
            exit 1
        fi
    fi
    python -m pip install --upgrade pip
    echo

    echo "Installing ffmpeg..."
    sudo apt update && sudo apt install -y ffmpeg
    if [ $? -ne 0 ]; then
        echo "Error installing ffmpeg. Check your system's package manager."
        exit 1
    fi

    echo "Installing Applio dependencies..."
    python -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error installing Applio dependencies."
        exit 1
    fi

    python -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --upgrade --index-url https://download.pytorch.org/whl/cu121 
    finish
}

# Function to finish installation
finish() {
    if [ -f "${requirements_file}" ]; then
        installed_packages=$(python -m pip freeze)
        while IFS= read -r package; do
            expr "${package}" : "^#.*" > /dev/null && continue
            package_name=$(echo "${package}" | sed 's/[<>=!].*//')
            if ! echo "${installed_packages}" | grep -q "${package_name}"; then
                echo "${package_name} not found. Attempting to install..."
                python -m pip install --upgrade "${package}"
            fi
        done < "${requirements_file}"
    else
        echo "${requirements_file} not found. Please ensure the requirements file with required packages exists."
        exit 1
    fi
    clear
    echo "Applio has been successfully downloaded. Run the file run-applio.sh to run the web interface!"
    exit 0
}

# Main menu loop
if [ "$(uname)" = "Darwin" ]; then
    if ! command -v brew >/dev/null 2>&1; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
        brew install python@3.10
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    fi
elif [ "$(uname)" != "Linux" ]; then
    echo "Unsupported operating system. Are you using Windows...?"
    echo "If yes, use the batch (.bat) file instead of this one!"
    exit 1
fi

prepare_install
