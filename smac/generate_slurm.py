
#usage
#generate-slurm.py problem algorithm array

problems = {
    "DTLZ": "1-7",
    "IMB": "1-14",
    "LZ09_F": "1-9",
    "MOP": "1-7",
    "UF": "1-9",
    "WFG": "1-9",
    "ZDT": "1-9"
}
algorithms = ["nsgaii", "nsgaiii", "moead", "omopso"]
#algorithsm = [ "smpso", "cmpso", "ibea"]

for algorithm in algorithms:
    for problem,array in problems.items():
        with open("{}-{}.slurm".format(algorithm, problem), "w") as slurm_file:
            slurm_file.write("#!/bin/bash\n")
            slurm_file.write("#SBATCH --array={}\n".format(array))
            slurm_file.write("#SBATCH --mem=4000\n")
            slurm_file.write("#SBATCH --ntasks=8\n")
            slurm_file.write("#SBATCH --nodes=1\n")
            slurm_file.write("#SBATCH --ntasks-per-node=8\n")
            slurm_file.write("#SBATCH --time=24:05:00\n\n")

            slurm_file.write("# Logging\n")
            slurm_file.write("#SBATCH --output=smac-{}-{}.out\n".format(algorithm, problem))
            slurm_file.write("#SBATCH --error=smac-{}-{}.err\n".format(algorithm, problem))
            slurm_file.write("#SBATCH --mail-type=ALL\n")
            slurm_file.write("#SBATCH --mail-user=sy6u19@soton.ac.uk\n\n")

            slurm_file.write("# load environment\n")
            slurm_file.write("module load python/3.7.3\n")
            slurm_file.write("source env/bin/activate\n\n")

            slurm_file.write("cd $HOME/cmlsga-jmetalpy\n")
            slurm_file.write("python src/tuning.py {} {}${SLURM_ARRAY_TASK_ID} ga 100000 5\n".format(algorithm, problem))

