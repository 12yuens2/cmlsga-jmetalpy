
#usage
#generate-slurm.py problem algorithm array

problems = {
    "DASCMOP": "1-6",
    "DTLZ": "1-7",
    "IMB": "1-14",
    "LZ09_F": "1-9",
    "MOP": "1-7",
    "UF": "1-9",
    "WFG": "1-9",
    "ZDT": "1-4,6"
}
algorithms = ["MOEAD", "U-NSGA-III", "OMOPSO", "CMPSO", "SMPSO", "cMLSGA", "U-NSGA-IIIe-cont", "U-NSGA-IIIe-non-cont", "MOEAD-e-cont"]#, "MOEAD-e-noncon"]

def write_slurm_file(filename, logfilename, running):
    with open(filename, "w") as slurm_file:
        slurm_file.write("#!/bin/bash\n")
        slurm_file.write("#SBATCH --array={}\n".format(array))
        slurm_file.write("#SBATCH --mem=8000\n")
        slurm_file.write("#SBATCH --ntasks=1\n")
        slurm_file.write("#SBATCH --nodes=1\n")
        slurm_file.write("#SBATCH --ntasks-per-node=1\n")
        slurm_file.write("#SBATCH --time=47:55:00\n\n")

        slurm_file.write("# Logging\n")
        slurm_file.write("#SBATCH --output=../logs/{}.out\n".format(logfilename))
        slurm_file.write("#SBATCH --error=../logs/{}.err\n".format(logfilename))
        slurm_file.write("#SBATCH --mail-type=FAIL\n")
        slurm_file.write("#SBATCH --mail-user=sy6u19@soton.ac.uk\n\n")

        slurm_file.write("# load environment\n")
        slurm_file.write("cd $HOME/cmlsga-jmetalpy\n")
        slurm_file.write("module load python/3.7.3\n")
        slurm_file.write("source env/bin/activate\n\n")

        slurm_file.write(running)

    print("Created {}".format(filename))

for algorithm in algorithms:
    for problem,array in problems.items():
        filename = "{}-{}.slurm".format(algorithm, problem)
        logfilename = "{}-{}".format(algorithm, problem)

        run_experiment = "python src/generate_visualisations.py /ssdfs/users/sy6u19/data-100000evals-30runs-corrections-normals/{}/{}${{SLURM_ARRAY_TASK_ID}} \n".format(algorithm, problem)

        if problem == "DASCMOP":
            for i in [5,6,7]:
                filename = "{}-{}-{}.slurm".format(algorithm, problem, i)
                logfilename = "{}-{}-{}".format(algorithm, problem, i)
                run_experiment = "python src/generate_visualisations.py /ssdfs/users/sy6u19/data-100000evals-30runs-corrections-normals/{}/{}${{SLURM_ARRAY_TASK_ID}}\({}\) \n".format(algorithm, problem, i)

                write_slurm_file(filename, logfilename, run_experiment)
        else:
            write_slurm_file(filename, logfilename, run_experiment)

