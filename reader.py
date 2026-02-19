def read_file(path):
    with open(path, 'r') as f:
        return f.read()

if __name__ == "__main__":
    content = read_file("example.txt")
    print(content)
