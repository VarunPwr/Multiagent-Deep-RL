# Multiagent Deep-Q learning
This code is implemented in Tensorflow-2 and Python 3.6
## Requirments
Latest version of Tensorflow-2 can be downloaded from the [official website](https://www.tensorflow.org/install). Tensorflow-2 requires [CUDA 11](https://developer.nvidia.com/cuda-11.0-download-archive)\
Install other dependencies using the following line of code 
```console
pip install --user -r requirements.txt
```
This package uses [openai/multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs), to install it use the following code
```console
pip install -e /mape/
```

## Getting started
For running a single agent DQN, use the following code*
```console
python main.py --net_mode 'DQN'
```
For multiagent indpendent DQN**,
```console
python main.py --net_mode 'DQN' --independent
```
For single agent DRQN, 
```console
python main.py --net_mode 'DQN' --recurrent
```
For single agent dueling,
```console
python main.py --net_mode 'Dueling' --recurrent
```
\* *Environment is set at multiagent by default. Change [this line](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/mape/multiagent/scenarios/simple_spread.py#L11) of /mape/multiagent/scenarios/simple_spread.py* \
\** *Multiagent DQN will use recurrent network by default*

## Instructions on modifying the game
### Changing environment
Import the desired environment class and change code [this line](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/main.py#L110) in *main.py* \

No further changes needed.
### Increasing agents

For increasing the number of agents follow as given below:
1. **/mape/multiagent/scenarios/simple_spread.py**: Change the number of agents in [this line](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/mape/multiagent/scenarios/simple_spread.py#L11)\

Other less relevant changes which can be made is reducing [agent](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/mape/multiagent/scenarios/simple_spread.py#L21) and [obstacle](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/mape/multiagent/scenarios/simple_spread.py#L25-L40) size as the evironment could clutter. 

2. **/deeprl_prj/idqn_keras.py**: Network size scales with increasing agents. Therefore it is necessary to modify [this line](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/deeprl_prj/idqn_keras.py#L58) and [this line](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/deeprl_prj/idqn_keras.py#L70), moreover the depth of policy layers can be increased in case of large policy size or number of agents. \
Add addtional networks in [these lines](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/deeprl_prj/idqn_keras.py#L166-L169). Subseqeuent changes are to be made in [calc_q_values](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/deeprl_prj/idqn_keras.py#L219-L231), [select_action](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/deeprl_prj/idqn_keras.py#L233-L254) and [update_policy function](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/deeprl_prj/idqn_keras.py#L256-L350). Correct [these lines](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/deeprl_prj/idqn_keras.py#L406-L407) depending on the policy size.


3. **/deeprl_prj/core.py**: Change the policy shape on [this line](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/deeprl_prj/core.py#L196).  A similar change is to be made in **/deeprl_prj/preprocesors.py**, [this line](https://github.com/VarunPwr/Multiagent-Deep-RL/blob/535174ac17a576f9e6fa9f682effe4c4cb4696b0/deeprl_prj/preprocessors.py#L31).
4. While running the code use the flag '-num_agents' which is set default at 2.

## Acknowledgements


