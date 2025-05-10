# MuJoCo 机械臂上位机部署

本项目用于机械臂控制系统上位机的部署。主要功能包括如下：
- 模拟和控制机械臂，基于 **MuJoCo** 进行物理仿真，支持 **键盘** 和 **Xbox 手柄** 等多种控制方式控制方式。
- 通过TCP/IP与其他网络设备进行通信
- 通过计算机设备的串口与下位机通过自定义协议进行通信
- 机械臂的控制包含轨迹跟随，运动学正逆解

本项目的实现代码依托于python多线程编程和mujoco仿真器

---

## 环境配置

### 安装 Anaconda

首先，为了确保计算机内python环境的独立性，使用者最好提前安装conda等python环境管理工具。安装 [Anaconda](https://www.anaconda.com/) 以便管理 Python 环境。安装完成后，使用以下命令创建环境并安装依赖。

### 配置Conda虚拟环境

请确保你已安装 **Anaconda**，然后在终端运行以下命令：

```bash
conda env create -f /config/environment.yaml #win环境下
conda env create -f /config/environment_linux.yaml #linux环境下
conda activate mujoco_env  # 进入环境
```
### 配置mujoco仿真器

除此之外最重要的环境是mujoco仿真器，已知[330（mujoco3.3.0）](https://github.com/google-deepmind/mujoco/releases)版本可稳定运行

---

## 运行仿真

确保环境正确配置后，可以尝试运行main文件夹下的main.py程序
**注意：** keyboard库在linux环境下必须使用sudo命令才能正常启动

### 项目文件树
    │  README.md        #说明配置文档
    │
    ├─.vscodevscode        #使用vscode时自动生成的配置文档
    │      settings.json
    │
    ├─config        #储存项目相关配置文件
    │      environment.yaml        #为win环境测试配置的conda配置文件
    │      environment_linux.yaml        #为linux环境配置的conda配置文件
    │      joint_names_cq_am_3.yaml
    │
    ├─history        #历史遗留文件（包含测试程序，旧的main程序）
    │  │  arm_axis_test.py        #可以单独运行的关节电机测试文件
    │  │  draw_figure.py        #用于绘制电机逆解情况
    │  │  main_v1.py
    │  │  main_v2_beta.py
    │  │  protocol.py        #第一版协议类
    │  │  tcp_client.py        #用于测试TCP通信，可单独在win运行
    │  │  tcp_server.py        #用于linux端单独测试服务器功能
    │  │
    │  └─__pycache__
    │          draw_figure.cpython-313.pyc
    │          protocol.cpython-313.pyc
    │
    ├─main        #此文件夹包含多个并行线程用于服务主线程工作
    │  │  Controller.py        #控制器类，包括键盘和手柄输入的处理
    │  │  DebugGUI.py        #测试类，无实际意义，可以直接删除
    │  │  Esp32Tcp.py        #用于与ESP32进行TCP通信的类
    │  │  main.py        #主程序
    │  │  main_backup.py        #测试类，无实际意义，可以直接删除
    │  │  Protocol.py        #用于与下位机串口通信的类
    │  │  SolveIK.py        #用于逆解算法的类
    │  │  Trajectory.py        #用于运动插补和轨迹生成的类
    │  │
    │  └─__pycache__
    │          Controller.cpython-313.pyc
    │          DebugGUI.cpython-313.pyc
    │          Protocol.cpython-313.pyc
    │          SolveIK.cpython-313.pyc
    │          Trajectory.cpython-313.pyc
    │
    ├─meshes        #模型文件夹
    │      base_link.STL
    │      link1.STL
    │      link2_0.STL
    │      link2_1.STL
    │      link2_2.STL
    │      link3.STL
    │      link4_1.STL
    │      link4_2.STL
    │      link5.STL
    │      link6.STL
    │      link7.STL
    │      link8.STL
    │
    └─urdf        #mujoco仿真时环境配置文件，主要关注cq_arm_3.xml和scene.xml,使用的是mujoco的mjcf格式。
            base_link.STL
            cq_arm_3.xml
            link1.STL
            link2_0.STL
            link2_1.STL
            link2_2.STL
            link3.STL
            link4_1.STL
            link4_2.STL
            link5.STL
            link6.STL
            link7.STL
            link8.STL
            scene.xml

### 主程序逻辑
##### main
1. 启动串口，TCP/IP,控制器，仿真器（主线程）
2. 进图循环，通过当前控制模式来决策
3. 退出后stop所有线程
##### Controller
1. 以固定频率处理来自键盘，手柄，自定义控制器的数据
2. 在main中被调用数据
##### Protocol
1. 在main中调用传递数据的函数将数据或是控制模式传递给Protocol的实例
2. 按50Hz固定频率将数据整合为自定义协议向下位机发送
##### Esp32Tcp
1. 启动后初始化目标端口作为监控目标（只能在配置了softAP模式的linux机上成功初始化）作为服务器。
2. 循环检测是否有设备连接该端口，如果有，就会监控设备向目标端口发送的信息
3. 按自定义协议对接收信息进行处理，传给Controller或者是轨迹的采样 
4. 如果已连接的设备中断，该线程会继续等待设备连接
##### SolveIK
1. 包含两种逆解方式
##### Trajectory
1. 包含两种轨迹生成方式

### 基本控制方式
    =============================
    机械臂控制器 基本使用说明
    =============================

    [ 键盘模式 按键控制 ]
    - 机械臂末端平移：
    W / S : 前进 / 后退
    A / D : 左移 / 右移
    Q / E : 上移 / 下移
    
    [ 手柄Xbox模式 摇杆和按键控制 ]
    - 机械臂末端平移：
    左摇杆Y轴: 前进 / 后退
    左摇杆X轴: 左移 / 右移
    右摇杆Y轴: 上移 / 下移
    
    [ 模式切换 ]
    - 空格键 : 切换输入模式（键盘 / 手柄）
    
    [ 注意事项 ]
    - 请确保手柄已连接并处于活动状态。
    - 程序启动时默认为手柄控制。
    - 程序启动时手柄未连接则切换到键盘控制。
    - 手柄在程序运行中连接,程序无法切换到手柄模式。
    - 其他操作请阅读Controller.py类
    - 当TCP/IP连接后,网络协议传入的数据也可以产生控制

---

## 备注

请确保 **MuJoCo** 界面处于 **激活状态**，否则键盘输入可能无效。
TCP/IP线程在win端一定初始化失败，在softAP模式下的linux机才能正常启动

---
© 2025 重庆机械臂项目

