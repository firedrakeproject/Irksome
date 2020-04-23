with open("jorge_data.txt") as f:
    lines = f.readlines()

# data file comes in 4-line blocks of the format
# <time>
# CD  <drag coeff>
# Turek Bencmkark CL <our lift coeff>
# Paper [2] CL <another lift coeff>

# let's munge that to
# Time, CD, CL1, CL2


with open("new_data.csv", "w") as f:
    f.write("# time, drag, lift1, lift2\n")
    for i in range(len(lines) // 4):
        t = float(lines[4 * i])
        CD = float(lines[4 * i + 1].split()[-1])
        CL1 = float(lines[4 * i + 2].split()[-1])
        CL2 = float(lines[4 * i + 3].split()[-1])
        f.write(",".join(map(str, [t, CD, CL1, CL2])))
        f.write("\n")

    
