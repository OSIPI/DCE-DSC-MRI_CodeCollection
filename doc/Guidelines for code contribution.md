# Guidelines for code contribution

*Note: The guidelines for code contribution will be updated if necessary, as we progress through different milestones of the OSIPI initiative. All queries regarding code submission can be submitted using the label 'question' on the [issue tracker](https://github.com/OSIPI/TF2.3-CodeLibrary/issues) or addressed to [Sudarshan Ragunathan](mailto:sudarshan.ragunathan@gmail.com).*

## Process for code contribution

We invite colleagues to contribute source code for performing one or more DCE-MRI or DSC-MRI processing steps. Currently, TF2.3 is accepting code written in Python. Code can be submitted directly as .py files by creating a feature branch of this repository (described below).

Contribution of the following is **optional** but helpful:
- .py test files
- test data
- files containing other relevant data or information, e.g. typical real-world applications of code, scripts, known features and limitations, citations etc.

If you would prefer to contribute via email or another mechanism, please contact the [task force leads](https://www.osipi.org/task-force-2-3/) or [Sudarshan Ragunathan](mailto:sudarshan.ragunathan@gmail.com).


## Essential contributor information

To help facilitate an organized approach to house community contributed source code for the different components of the DCE/DSC pipeline, we ask contributors to provide the following information **as part of the source code folder names**:

- Identify the component of the DCE/DCE pipeline that the code is intended to represent
- Name of the contributor's institution / organization
- Name of the contributor's lab / research group ( if applicable)
- Author of code (can also include in the source code) 

Further details can be submitted in a text file if necessary.

## How to submit code

Code submission can be performed directly to the repository by creating a feature branch containing the contributor information described above. A step by step guide is provided as follows:

##### Clone repo and switch to develop branch
	$git clone https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection.git
	$git switch develop
	$git status
Git status should say that you are now on branch develop.
##### Create new feature branch
	$git checkout -b <initials_featurebranchname>
	
Add contributions to feature branch (in folder src/original/'initials lab instution')
##### Push feature branch to remote
	$git add <all_untracked_files>
	$git commit -m "Commit Message"
	$git push -u origin <feature_branch_name>
	
##### In GitHub, make pull request from new feature branch to develop for approval
On github.com, select the **Pull Requests** tab and click on **New pull request**. 
![new pull request](images/pullrequest.png)
In the drop-down for **base** select the **develop** branch and under **compare** select the branch with your commits. Click on create pull request. 

 
