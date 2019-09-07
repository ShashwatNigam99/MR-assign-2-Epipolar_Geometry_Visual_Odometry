out = open('final.txt', 'w')
with open("output.txt") as file:
    for line in file:
        line = line.rstrip()
        if line:
            out.write(line + '\n')