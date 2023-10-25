with open('a.txt', 'r') as f:
    lines = f.readlines()
with open('a.txt', 'w') as f:
    for line in lines:
        if line.strip() and not line.startswith('#'):
            f.write(line)




