# The PolicyPrep: An automated workflow for experiment studies
## Background & Motivation :question:
Dear Researchers _(with a special shoutout to new Ph.D. students :wave:)_,

I am writing this to inform you of a new project I have been planning and working on for the past 
few months. It is called the PolicyPrep (i.e., a data pre-processing and model training pipeline). The purpose of this project is to automate the 
process of running experiments with deep learning frameworks such as PyTorch (although TensorFlow and others are possible). The pipeline is designed to be run 
on a remote machine, such as a server, and will run continuously until stopped. The pipeline will run experiments, collect the results, and train a policy using reinforcement learning
to be used in the next experiment. The pipeline will also generate a report of the results of the experiments
and the policy that was trained. 

**The PolicyPrep is not only for use with RL policy induction.** That is what its 
initially created for, but with a few minor tweaks, it can facilitate any experiment-related study, such as Inverse RL (this would be a matter of changing step 9 at this time of writing) or even posthoc analysis. It is meant to be a consistent and uniform platform for all lab members involved in your experiment's projects - at least, in the coming future, it will be.

Essentially, rather than each of us re-invent the wheel by collecting study data, aggregating 
it together, etc., the pipeline, at the very least, can perform these operations for you. 
This would involve running a smaller subset of the total steps in the pipeline or creating your own custom steps and appending those to it (e.g., make it offshoot into doing RL and Inverse RL simultaneously). **The whole purpose of this effort is to save your time so you can focus on other things, such as preparing your research question or having extra free time (yes, a Ph.D. student with free time** :exploding_head:). In the past, I have spent weeks or even months preparing my own local experiment setup, so your savings concerning time are significant. For a new Ph.D. student, a conservative estimate on time saved (as things are right now) amounts to at least 2 months throughout your entire Ph.D., as we have eliminated the need for looking up the data, costly edits, manually updating training data, or writing "fixes" to patch InferNet, as well as formatting the data automatically for you to use with RL (or Inverse RL, as previously stated).

It also serves as additional documentation on how we perform experiment setup for policies, guiding new students from start to finish on what we expect explicitly, so they can confidently continue their study knowing they have completed the necessary steps correctly.

Furthermore, by accepting the PolicyPrep in your workflow, it would be easier to assist you if you face trouble implementing your policy, as opposed to "you're on your own" if your code doesn't work. I hope establishing PolicyPrep will allow us to collaborate more closely and pursue research endeavors we otherwise might not have had the time or resources in the past.

Lastly, the pipeline is meant to be a living project, meaning it will be updated and improved 
over time. **Under all circumstances, do not hesitate to reach out to me if you have any 
questions or concerns**. I am more than happy to help you with any issues you may have, and I am 
open to any suggestions you may have to improve the pipeline. I am also open to any 
contributions you may have to the project. I am not the best programmer, so I am sure there are
many ways to improve the code. However, I am confident that the pipeline will be a useful tool
for all of us. 

Please do not let the pipeline intimidate you or be a barrier to your research. It is meant to be a tool to help you, not hinder 
you. I am here to help you with any issues you may have, and I am more than happy to do so. 

If you are interested in working together, please feel free to reach out to me regarding any questions you may have about incorporating the pipeline into your work (jwhostet@ncsu.edu). I would also appreciate any help offered to ensure the robustness of the project.

Sincerely, 

J. W. Hostetter
***
## Requirements :heavy_check_mark:
The following are required to run the pipeline:
- For packages, see `requirements.txt`
- Python 3.8.5
***
## Setup :hammer_and_wrench:
This project is written in Python 3.8.5. It is recommended to use a virtual environment to run this project.
The following instructions are for setting up the project on a Linux or macOS machine. The instructions may vary slightly
for other operating systems. 

Within PyCharm's markdown preview, the following bash commands can be executed by clicking on the _play_ button that appears
when hovering over the command.

1. Clone the repository:
    ```bash
    git clone https://github.com/johnHostetter/PolicyPrep.git
    ```
2. Change directory into the T directory:
    ```bash
    cd PolicyPrep
    ```
3. Create a virtual environment (within the PolicyPrep directory):
    ```bash
    python3 -m venv venv
    ```
4. Activate the virtual environment:
    ```bash
    source venv/bin/activate
    ```
5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
6. Pip install the project into an editable state, so that changes to the code will be reflected in the environment:
    ```bash
    pip install -e .
    ```
7. Verify step 6 executed correctly by running the following command(s):

   1. 
       ```bash
       pip list
       ```
       The output should contain the following line:
       ```bash
       PolicyPrep            1.0        /home/johnhostetter/PycharmProjects/PolicyPrep (or wherever the project is located)
       ```
      
   2. 
        ```bash
        pip show PolicyPrep
        ```
        The output should be:
        ```bash
        Name: PolicyPrep
        Version: 1.0 (or whatever version is currently installed)
        Summary: UNKNOWN
        Home-page: UNKNOWN
        Author: John Wesley Hostetter
        Author-email: jwhostet@ncsu.edu
        License: UNKNOWN
        Location: /Users/jwhostet/PolicyPrep (or wherever the project is located)
        Requires:
        Required-by:
        ```
      
   3. 
        ```bash
        pip freeze
        ```
        The output should contain the following line:
        ```bash
        -e git+https://github.com/johnHostetter/PolicyPrep@5856f3b709750d5a6aa7403f8c0c5668873fbd5f#egg=PolicyPrep
        ```
   
8. Run the pipeline (two ways):
   1. In the foreground (good for testing):
       ```bash
       python3 src/pipeline.py
       ```
    2. In the background (good for running the pipeline for a long time):
         ```bash
       nohup python3 -u src/pipeline.py &
       ```
         After running this command, you may need to hit _ENTER_ on your keyboard. The process 
       is now being executed _in the background_, meaning you can perform other commands in the 
       terminal, or close the terminal (e.g., ssh disconnect). The output to the terminal will be 
       written to a file called _nohup.out_ in the PolicyPrep directory. View the contents with the following command:
       ```bash
       vi nohup.out
       ```
       Move to the end of the file with _:$_ and type _:q_ to exit the file. Be sure to view 
       this file is being updated periodically to ensure the pipeline is running correctly. I 
       recommend every few hours (before training of InferNet) to ensure the pipeline is running.
       Once you reach the training of InferNet, I recommend checking the file once every day or two.
   3. To stop the pipeline (if running in the foreground), press _CTRL + C_.
   4. To stop the pipeline (if running in the background), run the following command:
       ```bash
       ps aux | grep pipeline.py
       ```
       The output will be similar to the following:
       ```bash
       jwhostet  12345  0.0  0.0  12345  1234 pts/0    S+   00:00   0:00 python3 src/pipeline.py
       jwhostet  12346  0.0  0.0  12345  1234 pts/0    S+   00:00   0:00 grep --color=auto pipeline.py
       ```
       The first number in the second column is the process ID (PID) of the pipeline. To stop the pipeline, run the following command:
       ```bash
       kill -9 12345
       ```
       where 12345 is the PID of the pipeline. You can verify the pipeline has stopped by running the following command:
       ```bash
       ps aux | grep pipeline.py
       ```
       An alternative to the above command to find the process ID is to run the following command:
       ```bash
       ps -ef | grep pipeline.py
       ```
9. The output will be generated in the _data_ folder, underneath the subdirectory called _for_policy_induction_. 
Within this subdirectory, there will be two folders: _pandas_ and _d3rlpy_, containing .csv files or .h5 files for 
policy induction via reinforcement learning, respectively.
***
## Troubleshooting :worried:
1. Initially when the pipeline.py is being run on the macOS (with a M1 Processor), a problem is being encountered while building up the wheel for lxml:

   > Building wheel for lxml (setup.py) ... error
   
   > **error**: **subprocess-exited-with-error**
   
   > **ERROR: Failed building wheel for lxml**

   Looking more into the issue, Adittya Soukarjya Saha came across this: 
   
   > Pip install on macOS gets error: command '/usr/bin/clang' failed with exit code 1
   
   _**FIX**:_ Updating the "_setup tools_":
   
   ```bash
   python3 -m pip install --upgrade setuptools
   ```
2. List of all the severe warnings(14) that were generated while running the code are given below. 
   Some of the requirements from the list provided in the requirements.txt file can be ignored as the code ran without using any of those listed, just fine.
* ```bash
  'nvidia-cublas-cu11 11.10.3.66' is not installed (required: 11.10.3.66, installed: <nothing>, latest: 11.11.3.6)
  ```
* ```bash
  'nvidia-cuda-cupti-cu11 11.7.101' is not installed (required: 11.7.101, installed: <nothing>, latest: 11.8.87)
  ```
* ```bash
  'nvidia-cuda-nvrtc-cu11 11.7.99' is not installed (required: 11.7.99, installed: <nothing>, latest: 11.8.89)
  ```
* ```bash
  'nvidia-cuda-runtime-cu11 11.7.99' is not installed (required: 11.7.99, installed: <nothing>, latest: 11.8.89)
  ```
* ```bash
  'nvidia-cudnn-cu11 8.5.0.96' is not installed (required: 8.5.0.96, installed: <nothing>, latest: 8.9.2.26)
  ```
* ```bash
  'nvidia-cufft-cu11 10.9.0.58' is not installed (required: 10.9.0.58, installed: <nothing>, latest: 10.9.0.58)
  ```
* ```bash
  'nvidia-curand-cu11 10.2.10.91' is not installed (required: 10.2.10.91, installed: <nothing>, latest: 10.3.0.86)
  ```
* ```bash
  'nvidia-cusolver-cu11 11.4.0.1' is not installed (required: 11.4.0.1, installed: <nothing>, latest: 11.4.1.48)
  ```
* ```bash
   'nvidia-cusparse-cu11 11.7.4.91' is not installed (required: 11.7.4.91, installed: <nothing>, latest: 11.7.5.86)
  ```  
* ```bash
   'nvidia-nccl-cu11 2.14.3' is not installed (required: 2.14.3, installed: <nothing>, latest: 2.18.3)
  ```  
* ```bash
   'nvidia-nvtx-cu11 11.7.91' is not installed (required: 11.7.91, installed: <nothing>, latest: 11.8.86)
  ```  
* ```bash
   'tensorflow-io-gcs-filesystem 0.32.0' is not installed (required: 0.32.0, installed: <nothing>, latest: 0.32.0)
  ```  
* ```bash
   'triton 2.0.0' is not installed (required: 2.0.0, installed: <nothing>, latest: 2.0.0.post1)
  ```  
* ```bash
   'tensorflow 2.12.0' is not installed (required: 2.12.0, installed: 2.13.0, latest: 2.13.0)
  ```  

The requirements that can be ignored are given below: 

1. [x] nvidia-cublas-cu11==11.10.3.66
2. [x] nvidia-cuda-cupti-cu11==11.7.101
3. [x] nvidia-cuda-nvrtc-cu11==11.7.99
4. [x] nvidia-cuda-runtime-cu11==11.7.99
5. [x] nvidia-cudnn-cu11==8.5.0.96
6. [x] nvidia-cufft-cu11==10.9.0.58
7. [x] nvidia-curand-cu11==10.2.10.91
8. [x] nvidia-cusolver-cu11==11.4.0.1
9. [x] nvidia-cusparse-cu11==11.7.4.91
10. [x] nvidia-nccl-cu11==2.14.3
11. [x] nvidia-nvtx-cu11==11.7.91
12. [x] tensorflow-io-gcs-filesystem==0.32.0
13. [x] triton==2.0.0

There were some more weak warnings, but those can be ignored.       
