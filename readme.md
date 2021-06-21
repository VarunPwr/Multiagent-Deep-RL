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
\* *Environment is set at multiagent by default. Change line 11 of /mape/multiagent/scenarios/simple_spread.py* \
\** *Multiagent DQN will use recurrent network by default*

## Instructions on modifying the game
### Changing environment
Add the preferred environment class and change code in lines 110 and 129 in *main.py* \
```python
    env = MultiagentEnv(args.num_agents)
    dqn.fit(env, args.num_samples, args.max_episode_length)

```
No further changes needed.
### Increasing agents

For increasing the number of agents follow as given below:
1. **/mape/multiagent/scenarios/simple_spread.py**: Change the number of agents in line 11
```
self.num_agents = 2
```
Other less relevant changes which can be made is reducing agent and obstacle size as the evironment could clutter. These changes can be made in lines 21, 28, 33 and 40.

2. **/deeprl_prj/idqn_keras.py**: Network size scales with increasing agents. Therefore it is necessary to modify lines 58 and 70, moreover the depth of policy layers can be increased in case of large policy size or number of agents. \
Add addtional networks in following the lines of code 166-169. Subseqeuent changes are to be made in calc_q_values, select_action and update_policy functions. Correct lines 406-407 depending on the policy size.


3. **/deeprl_prj/core.py**: Change the policy shape on line 196.  A similar change is to be made in **/deeprl_prj/preprocesors.py**, line 31.
4. While running the code use the flag '-num_agents' which is set default at 2.

## Acknowledgements


