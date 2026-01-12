<h2 align="center">
  <b>CalliRewrite: Recovering Handwriting Behaviors from Calligraphy Images without Supervision</b>

  <b><i>ICRA 2024</i></b>


<div align="center">
    <a href="https://arxiv.org/abs/2405.15776" target="_blank">
    <img src="https://img.shields.io/badge/Paper-ArXiv-red" alt="Paper arXiv"></a>
    <a href="https://luoprojectpage.github.io/callirewrite/" target="_blank">
    <img src="https://img.shields.io/badge/Page-CalliRewrite-blue" alt="Project Page"/></a>
    <a href="http://igcl.pku.edu.cn/papers/24/ICRA2024_CalliRewrite_LYX.pdf" target="_blank">
    <img src="https://img.shields.io/badge/Lab-Link-green" alt="Lab Link"></a>
</div>
</h2>

This is the repository of [**CalliRewrite: Recovering Handwriting Behaviors from Calligraphy Images without Supervision**](http://igcl.pku.edu.cn/papers/24/ICRA2024_CalliRewrite_LYX.pdf).

CalliRewrite is an unsupervised approach enabling low-cost robotic arms to replicate diverse calligraphic 
glyphs by manipulating different writing tools. We use fine-tuned unsupervised LSTM to perform coarse stroke segmentation, and refine them through a reinforcement learning method to produce 
fine-grained control.

For more information, please visit our [**project page**](https://luoprojectpage.github.io/callirewrite/).

![CalliRewrite Teaser](demo/teaser.png)


## 📬 News
- **2026.1** 🎉 Added MuJoCo simulation support and Franka robot integration!
- **2026.1** 📚 Comprehensive documentation added for all modules
- **2024.4.25** Coarse Sequence Extraction checkpoints released.
- **2024.4.6** CalliRewrite has been selected as a finalist for the IEEE ICRA Best Paper Award in Service Robotics
- **2024.2.27** Version 1.0 upload

## How to Use Our Code and Model:
We are releasing our network and checkpoints. The weights for the coarse sequence module are stored in [**this repository**](https://drive.google.com/file/d/1PUghb8WizEOYHYIAdBluwQMbTeRlBqF1/view?usp=sharing). You can download and place them in the folder ./outputs/snapshot/new_train_phase_1 and ./outputs/snapshot/new_train_phase_2. The parameters of the SAC model are learned based on the provided textual images; therefore, they vary depending on the different tools and input texts."

You can setup the pipeline under the following guidance.

### 0. Install dependencies
Due to package version dependencies, we need to set up two separate environments for coarse sequence extraction and tool-aware finetuning. To do this, you can follow these steps:

1. Navigate to the directory for coarse sequence extraction:
   ```bash
   cd seq_extract
   ```

2. Create a new conda environment using the specified requirements file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the newly created environment (optional):
   ```bash
   conda activate calli_ext
   ```
   
4. Navigate to the directory for sequence fine-tuning:
   ```bash
   cd ../rl_finetune
   ```

5. Create another new conda environment using the requirements file for fine-tuning:
   ```bash
   conda env create -f environment.yml
   ```

6. Activate the second newly created environment:
   ```bash
   conda activate callli_rl
   ```
   
7. Follow **modify_env.md** and correct some flaws in the packages (Necessary!):
   

By following these steps, you will have two separate conda environments configured for coarse sequence extraction and sequence fine-tuning, ensuring that the correct dependencies are installed for each task.

## 📖 Documentation

We provide comprehensive documentation for all modules:

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture and data flow
- **[seq_extract/TRAINING_PROCESS.md](seq_extract/TRAINING_PROCESS.md)** - Detailed LSTM training process
- **[rl_finetune/TRAINING_PROCESS.md](rl_finetune/TRAINING_PROCESS.md)** - SAC reinforcement learning training
- **[callibrate/NPZ_FILE_EXPLAINED.md](callibrate/NPZ_FILE_EXPLAINED.md)** - NPZ file format and conversion
- **[callibrate/REAL_CASE_STUDY.md](callibrate/REAL_CASE_STUDY.md)** - Real case study walkthrough
- **[callibrate/FRANKA_SETUP.md](callibrate/FRANKA_SETUP.md)** - Franka robot setup guide
- **[mujoco_sim/README.md](mujoco_sim/README.md)** - MuJoCo simulation guide


### 1. Caliberate your own writing utensil

We provide three simple tools for modeling in the reinforcement learning environment: **Calligraphy brush**, **fude pen**, and **flat tip marker**. The geometry and dynamic properties are defined in `./rl_finetune/Callienv/envs/tools.py` and folder `./rl_finetune/tool_property/`.  You can also define your own utensil easily.


For robotic demonstration, we provide a jupyter notebook in `./callibrate/callibrate.ipynb`. It clearly demonstrates the whole process to find out the $r-z$ correspondence on Dobot Magician robotic arm. Remember this step is crucial for it bridges the sim2real gap and may affect the rewriting result.

### 2. Download pretrained models

### 3. Coarse Sequence Extraction

#### Training
We conduct a two-phase progressive training, for the first phase we train on QuickDraw Dataset, you can simply run the shell command:
   ```bash
   conda activate calli_ext
   cd seq_extract
   ```
Then you can train the two-phase model with your owncollected fine-tuned data:
   ```bash
   bash ./train.sh
   ```

#### Testing with Trained Models
For example, test the model saved in "outputs/snapshot/new_train_phase_2" on images in the "imgs" folder:

 ```bash
 python ./test.py --input imgs --model new_train_phase_2
 ```

### 4. Tool-Aware Finetuning

You can call the function to easily move inferenced sequences along with the images to `./rl_finetune/data/train_data' and `./rl_finetune/data/test_data'. Remember the number of train/test envs must be a divisor of the number of train/test data.
   ```bash
   conda activate calli_rl
   cd ..
   python move_data
   cd rl_finetune
   ```
Then you can have an easy startup:
   ```bash
   bash ./scripts/train_brush.sh
   ```

### 5. MuJoCo Simulation (Optional but Recommended)

Before deploying to real robots, you can test trajectories in MuJoCo simulation:

```bash
# Install MuJoCo dependencies
cd mujoco_sim
pip install -r requirements.txt

# Quick test
bash quick_test.sh

# Run simulation with your trajectory
python mujoco_simulator.py ../callibrate/examples/example_永.npz --speed 0.05

# Record video
python mujoco_simulator.py ../callibrate/examples/example_永.npz \
    --record outputs/video.mp4 --speed 0.05
```

See [mujoco_sim/README.md](mujoco_sim/README.md) for detailed documentation.

### 6. Robot Deployment

#### For Dobot Magician (Original)

Follow the calibration notebook in `./callibrate/callibrate.ipynb`.

#### For Franka Emika (Panda/FR3) - NEW!

We now support Franka robots with comprehensive calibration and control:

```bash
cd callibrate

# 1. Generate calibration test file
python calibrate.py --mode generate --tool brush

# 2. Run calibration on robot
python RoboControl.py examples/test_calibration.npz <robot_ip> 0.05

# 3. Measure and fit r-z relationship
python calibrate.py --mode fit --tool brush \
    --widths <measured_widths> --zs <test_heights>

# 4. Convert RL output to robot coordinates
python calibrate.py --mode convert --tool brush \
    --input ../rl_finetune/results/character.npy \
    --output character.npz --alpha 0.04 --beta 0.5

# 5. Execute on robot
python RoboControl.py character.npz <robot_ip> 0.05
```

See [callibrate/FRANKA_SETUP.md](callibrate/FRANKA_SETUP.md) for detailed setup instructions.

### 7. Visualization and Analysis


## Citation
```
@article{luo2024callirewrite,
  title={CalliRewrite: Recovering Handwriting Behaviors from Calligraphy Images without Supervision},
  author={Luo, Yuxuan and Wu, Zekun and Lian, Zhouhui},
  journal={arXiv preprint arXiv:2405.15776},
  year={2024}
}
```
