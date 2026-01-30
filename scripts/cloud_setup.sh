#!/bin/bash
# Cloud instance setup script for MD data generation
# Supports AWS, GCP, and Azure
# Run this on your cloud instance after launching

set -e

echo "========================================="
echo "MD Data Generation - Cloud Setup"
echo "========================================="

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect OS"
    exit 1
fi

echo "Detected OS: $OS"

# Install dependencies
echo ""
echo "Installing dependencies..."

if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        python3 \
        python3-pip \
        libopenmpi-dev \
        openmpi-bin \
        libfftw3-dev \
        libjpeg-dev \
        libpng-dev \
        python3-numpy \
        python3-matplotlib
        
elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ]; then
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y \
        cmake \
        git \
        wget \
        python3 \
        python3-pip \
        openmpi-devel \
        fftw-devel \
        libjpeg-devel \
        libpng-devel
else
    echo "Unsupported OS: $OS"
    exit 1
fi

# Install Python packages
echo ""
echo "Installing Python packages..."
pip3 install --upgrade pip
pip3 install numpy scipy matplotlib h5py

# Install LAMMPS
echo ""
echo "Installing LAMMPS..."

LAMMPS_VERSION="stable_29Aug2024"
LAMMPS_DIR="$HOME/lammps"

if [ ! -d "$LAMMPS_DIR" ]; then
    cd $HOME
    git clone -b $LAMMPS_VERSION https://github.com/lammps/lammps.git
    cd lammps
    mkdir build
    cd build
    
    # Configure with MPI and common packages
    cmake ../cmake \
        -DCMAKE_INSTALL_PREFIX=$HOME/lammps-install \
        -DBUILD_MPI=yes \
        -DPKG_MOLECULE=yes \
        -DPKG_MANYBODY=yes \
        -DPKG_KSPACE=yes \
        -DPKG_RIGID=yes \
        -DPKG_EXTRA-DUMP=yes
    
    # Build (use all available cores)
    make -j$(nproc)
    make install
    
    # Add to PATH
    echo "export PATH=\$PATH:$HOME/lammps-install/bin" >> ~/.bashrc
    export PATH=$PATH:$HOME/lammps-install/bin
    
    echo "✓ LAMMPS installed successfully"
else
    echo "✓ LAMMPS already installed"
fi

# Verify installation
echo ""
echo "Verifying installation..."
if command -v lmp &> /dev/null; then
    echo "✓ LAMMPS executable found: $(which lmp)"
    lmp -help | head -n 5
else
    echo "✗ LAMMPS not found in PATH"
    echo "  Add to PATH: export PATH=\$PATH:$HOME/lammps-install/bin"
fi

# Clone/setup project
echo ""
echo "Setting up project..."
PROJECT_DIR="$HOME/msnns-4-dense-matter"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Please clone your project repository to $PROJECT_DIR"
    echo "  git clone <your-repo-url> $PROJECT_DIR"
else
    echo "✓ Project directory found: $PROJECT_DIR"
    cd $PROJECT_DIR
    
    # Install Python requirements
    if [ -f requirements.txt ]; then
        pip3 install -r requirements.txt
        echo "✓ Python requirements installed"
    fi
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_DIR"
echo "2. Generate MD inputs: python scripts/generate_md_data.py --num_configs 1000"
echo "3. Run simulations: ./data/md_simulations/run_all.sh"
echo "4. Monitor progress: tail -f data/md_simulations/*/log.lammps"
echo ""
echo "Estimated time for 1000 configs: 2-3 days on 16-core instance"
echo "========================================="
