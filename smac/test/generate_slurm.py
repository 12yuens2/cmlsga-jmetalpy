
#usage
#generate-slurm.py problem algorithm array

problems = {
    "DASCMOP": "1-6"
    #"DTLZ": "1-7",
    #"IMB": "1-14",
    #"LZ09_F": "1-9",
    #"MOP": "1-7",
    #"UF": "1-9",
    #"WFG": "1-9",
    #"ZDT": "1-6"
}
algorithms = ["nsgaii", "nsgaiii", "moead", "ibea", "smpso", "cmpso", "omopso"]

for algorithm in algorithms:
    for problem,array in problems.items():
        filename = "{}-{}-7.slurm".format(algorithm, problem)
        with open(filename, "w") as slurm_file:
            slurm_file.write("#!/bin/bash\n")
            slurm_file.write("#SBATCH --array={}\n".format(array))
            slurm_file.write("#SBATCH --mem=3000\n")
            slurm_file.write("#SBATCH --ntasks=1\n")
            slurm_file.write("#SBATCH --nodes=1\n")
            slurm_file.write("#SBATCH --ntasks-per-node=1\n")
            slurm_file.write("#SBATCH --time=15:05:00\n\n")

            slurm_file.write("# Logging\n")
            slurm_file.write("#SBATCH --output=/scratch/sy6u19/batch-output/smac-{}-{}-7.out\n".format(algorithm, problem))
            slurm_file.write("#SBATCH --error=/scratch/sy6u19/batch-output/smac-{}-{}-7.err\n".format(algorithm, problem))
            slurm_file.write("#SBATCH --mail-type=ALL\n")
            slurm_file.write("#SBATCH --mail-user=sy6u19@soton.ac.uk\n\n")

            slurm_file.write("# load environment\n")
            slurm_file.write("cd $HOME/cmlsga-jmetalpy\n")
            slurm_file.write("module load python/3.7.3\n")
            slurm_file.write("source env/bin/activate\n\n")

            algo_type = "ga"
            if "pso" in algorithm:
                algo_type = "pso"
            slurm_file.write("python src/tuning.py {} {}${{SLURM_ARRAY_TASK_ID}}\(7\) {} 100000 5\n".format(algorithm, problem, algo_type))

        print("Created {}".format(filename))

