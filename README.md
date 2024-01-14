# Real-time-Path-planning-with-SEPSO
This is the code of the paper [Efficient Real-time Path Planning with Self-evolving Particle Swarm Optimization in Dynamic Scenarios](https://arxiv.org/abs/2308.10169). 

  |                           Static Path Planning with DTPSO                           |                         Dynamic Path Planning with SEPSO                          |
  | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="https://github.com/XinJingHao/Real-time-Path-planning-with-SEPSO/blob/main/Code_for_Path_Plannning/Static%20Path%20Planning%20with%20DTPSO/static.gif" width="320" height="320"> | <img src="https://github.com/XinJingHao/Real-time-Path-planning-with-SEPSO/blob/main/Code_for_Path_Plannning/Dynamic%20Path%20Planning%20with%20SEPSO(Application%20Phrase)/dynamic.gif" width="320" height="320"> |

**Important Note:** We have published the [OkayPlan](https://github.com/XinJingHao/OkayPlan), a new generation of SEPSO. 

## 1. Dependencies 
```python
torch==2.0.1
numpy==1.24.3
pygame==2.1.3
tensorboard==2.13.0
python==3.10.3
```

## 2. How to use my code
### 2.1 For 'Dynamic Path Planning with SEPSO':
```bash
cd Code_for_Path_Plannning
cd Dynamic Path Planning with SEPSO(Application Phrase)
python SEPSO_Path_AP_main.py
```
These commands will run the code on CPU. If your device is compatible with GPU, you can replace '**python SEPSO_Path_AP_main.py**' with '**python SEPSO_Path_AP_main.py --dvc cuda**' to expedite the running.

### 2.2 For 'Static Path Planning with DTPSO':
```bash
cd Code_for_Path_Plannning
cd Static Path Planning with DTPSO
python DTPSO_Path_static.py
```
These commands will run the code on CPU. If your device is compatible with GPU, you can replace '**python DTPSO_Path_static.py**' with '**python DTPSO_Path_static.py --dvc cuda**' to expedite the running.


## 3. Citing the Project

To cite this repository in publications:

```bibtex
@misc{SEPSO2023JinghaoXin,
      title={Efficient Real-time Path Planning with Self-evolving Particle Swarm Optimization in Dynamic Scenarios}, 
      author={Jinghao Xin and Zhi Li and Yang Zhang and Ning Li},
      year={2023},
      eprint={2308.10169},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
## Maintenance History
+ 2023/10/16
  + **_Real-time-Path-planning-with-SEPSO/Code_for_Path_Plannning/Dynamic Path Planning with SEPSO(Application Phrase)/SEPSO_Path_AP_main.py_** is upgraded:
    + obstacle velocity can be rendered now
    + add shortcut recording option
    + the corresponding 'Obstacle_Segments.pt' is upgraded to avoid oscillation bug

