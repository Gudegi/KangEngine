import sys
sys.path.append("./build")
import kangengine


if __name__ == "__main__":
    print(kangengine.add(1,3))
    app = kangengine.App()
    app.initialize(3840, 2160, False)
    app.start()
