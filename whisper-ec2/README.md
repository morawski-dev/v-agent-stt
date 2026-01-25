# Running Whisper Large V3 on AWS EC2 with GPU

## Step-by-step guide for audio transcription

---

## PART 1: Creating and configuring EC2 instance

### Step 1: Log into AWS Console
1. Log into [AWS Management Console](https://console.aws.amazon.com)
2. Navigate to **EC2** service
3. Select region (e.g., us-east-1 or eu-central-1)

### Step 2: Check GPU limits
 **IMPORTANT**: Before creating an instance, check limits:
1. In EC2 console → **Limits** (in left menu)
2. Check limit for **G** type instances (GPU)
3. If limit = 0, you need to request a limit increase:
    - Service Quotas → AWS Services → Amazon EC2
    - Find "Running On-Demand G and VT instances"
    - Request quota increase

### Step 3: Launch EC2 instance

1. **Click "Launch Instance"**

2. **Choose name:**
    - Name: `whisper-gpu-server`

3. **Select AMI (system image):**
    - Select: **Deep Learning AMI GPU PyTorch 2.1.0 (Ubuntu 20.04)**
    - Alternatively: **Ubuntu Server 22.04 LTS** (but requires more installation)

4. **Select instance type:**
    - **Recommended**: `g5.xlarge` (4 vCPU, 16GB RAM, NVIDIA A10G, ~$1.00/h)
    - Cheaper: `g4dn.xlarge` (4 vCPU, 16GB RAM, NVIDIA T4, ~$0.50/h)
    - Faster: `g5.2xlarge` (8 vCPU, 32GB RAM, NVIDIA A10G, ~$1.20/h)

5. **Key pair:**
    - Create new key pair if you don't have one
    - Name: `whisper-key`
    - Type: RSA
    - Format: `.pem` (Linux/Mac) or `.ppk` (Windows/PuTTY)
    - **Download and save** the key file!

6. **Network settings:**
    - Create security group:
        - Name: `whisper-sg`
        - Description: `Security group for Whisper server`
        - Rules:
            - **SSH (22)**: Source = My IP (Your IP)
            - Optionally **HTTP (80)** and **HTTPS (443)** if you want API

7. **Storage (disk):**
    - Increase to **100 GB gp3** (model is large, ~6GB + dependencies)

8. **Click "Launch Instance"**

### Step 4: Connect to instance

Wait 2-3 minutes for the instance to start.

**Linux/Mac:**
```bash
# Set key permissions
chmod 400 whisper-key.pem

# Connect
ssh -i whisper-key.pem ubuntu@<PUBLIC_IP>
```

**Windows (PowerShell):**
```powershell
ssh -i whisper-key.pem ubuntu@<PUBLIC_IP>
```

Replace `<PUBLIC_IP>` with the Public IPv4 address of your instance (found in EC2 console).

---

## PART 2: Software installation

### Step 5: Update system

```bash
# Update packages
sudo apt update
sudo apt upgrade -y
```

### Step 6: Install NVIDIA Drivers (if not using Deep Learning AMI)

**Check if drivers are already installed:**
```bash
nvidia-smi
```

If you see GPU information - skip to Step 7.

**If there are no drivers:**
```bash
# Install NVIDIA drivers
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# Reboot
sudo reboot
```

After reboot, reconnect and check:
```bash
nvidia-smi
```

### Step 7: Install CUDA Toolkit (if needed)

Check CUDA version:
```bash
nvcc --version
```

If there's no CUDA:
```bash
# Download CUDA 12.1 (compatible with PyTorch)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 8: Install Python and pip

```bash
# Check Python
python3 --version  # Should be version 3.8+

# Install pip
sudo apt install -y python3-pip python3-venv

# Install ffmpeg (required by Whisper)
sudo apt install -y ffmpeg
```

### Step 9: Create virtual environment

```bash
# Create project directory
mkdir ~/whisper-project
cd ~/whisper-project

# Create virtual environment
python3 -m venv whisper-env

# Activate
source whisper-env/bin/activate
```

### Step 10: Install PyTorch with GPU support

```bash
# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check if GPU is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

You should see:
```
CUDA available: True
GPU: NVIDIA A10G (or other GPU model)
```

### Step 11: Install Whisper and dependencies

```bash
# Install transformers (Hugging Face)
pip install transformers accelerate

# Install additional libraries
pip install soundfile librosa
```

---

## PART 3: Preparing audio file and script

### Step 12: Prepare test audio file

**Option A: Use sample file from internet**
```bash
# Download sample audio file (English)
wget https://www2.cs.uic.edu/~i101/SoundFiles/taunt.wav -O test_audio.wav
```

**Option B: Upload your own file**
```bash
# From your computer (from new terminal):
scp -i whisper-key.pem your_file.wav ubuntu@<PUBLIC_IP>:~/whisper-project/test_audio.wav
```

### Step 13: Transcription script

```bash
transcribe.py
```

Set permissions:
```bash
chmod +x transcribe.py
```

---

## PART 4: Running transcription

### Step 14: First transcription

```bash
# Make sure virtual environment is active
source whisper-env/bin/activate

# Run transcription
python3 transcribe.py test_audio.wav
```

**What will happen:**
1. On first run, model will be downloaded (~3GB) - this will take 5-10 minutes
2. Model will be loaded onto GPU
3. Audio file will be transcribed
4. Result will be displayed and saved to `.txt` file

**If you have a file in Polish:**
```bash
python3 transcribe.py test_audio.wav pl
```

### Step 15: Check GPU usage

During transcription, in a new terminal window:
```bash
# Connect again to EC2
ssh -i whisper-key.pem ubuntu@<PUBLIC_IP>

# Monitor GPU
watch -n 1 nvidia-smi
```

You should see GPU and memory usage.

---

## PART 5: Batch script for multiple files

### Step 16: Transcribe multiple files

Usage:
```bash
# Create directory with audio files
mkdir audio_files
# ... move your files there ...

# Transcribe everything
python3 batch_transcribe.py audio_files pl
```

---

## PART 6: Optimization and tips

### Performance tips

1. **Model size vs accuracy:**
    - `whisper-large-v3`: Best quality, requires more GPU
    - `whisper-medium`: Good compromise
    - `whisper-small`: Faster, lower accuracy

2. **Batch processing:**
    - Parameter `batch_size=16` in pipeline can be increased if you have more GPU memory

3. **Long recordings:**
    - `chunk_length_s=30` splits audio into 30-second chunks
    - Can be reduced if you have low GPU memory

4. **Precision:**
    - `float16` (used by default on GPU) is faster
    - `float32` is more accurate but slower

### Cost monitoring

```bash
# Check instance uptime
uptime

# Remember to stop instance when not using!
```

In AWS Console → EC2 → Select Instance → Instance State → **Stop**

**IMPORTANT**: GPU instances are expensive! Stop them when not using.

---

## PART 7: Troubleshooting

### Problem: CUDA out of memory

**Solution:**
```python
# In transcribe.py change:
batch_size=8  # instead of 16
chunk_length_s=15  # instead of 30
```

### Problem: Model downloads too slowly

**Solution:**
```bash
# Set cache directory to disk with more space
export HF_HOME=/home/ubuntu/whisper-project/.cache
```

### Problem: Low transcription quality

**Solution:**
1. Specify language explicitly: `python3 transcribe.py file.wav pl`
2. Use better audio quality (16kHz or higher)
3. Check if audio is not distorted

### Problem: "Permission denied" with SSH

**Solution:**
```bash
chmod 400 whisper-key.pem
```

---

## PART 8: Cleanup and shutdown

### Stop instance

**IMPORTANT**: GPU instances are expensive (~$1/h). Stop them when not using!

**Through AWS console:**
1. EC2 → Instances
2. Select instance
3. Instance state → Stop

**From terminal:**
```bash
# Log in and stop
sudo shutdown -h now
```

### Delete resources (if you finished the project)

1. EC2 → Instances → Terminate instance
2. EC2 → Volumes → Delete volume (if not automatically deleted)
3. EC2 → Security Groups → Delete security group

---

## BONUS: Quick test script

```bash
python3 quick_test.py
```

---

## Summary

Now you have a complete environment for audio transcription on AWS EC2 with GPU!

**Key commands:**
```bash
# Connection
ssh -i whisper-key.pem ubuntu@<PUBLIC_IP>

# Activate environment
cd ~/whisper-project
source whisper-env/bin/activate

# Transcription
python3 transcribe.py file.wav [language]

# Monitor GPU
nvidia-smi

# Stop
sudo shutdown -h now
```

**Estimated costs (us-east-1):**
- g5.xlarge: ~$1.00/h
- g4dn.xlarge: ~$0.50/h
- Storage (100GB): ~$10/month

**Pro tip**: Use Spot Instances for 70% savings (Instance settings → Request Spot instances)
