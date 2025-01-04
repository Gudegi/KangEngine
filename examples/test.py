import sys
import os
sys.path.append(os.getcwd() + "/build")
import kangengine


class Myengine:
    def __init__(self, width, height):
        self.engine = kangengine.App()
        self.engine.initialize(width, height, False)

    def pre_render():
        pass

    def render():
        pass

    def run(self):
        self.engine.step()
        frame_count = 0
        while True:
            frame_count += 1
            self.engine.step()
        return



if __name__ == "__main__":
    print(kangengine.add(1,3))
    app = kangengine.App()
    app.initialize(1920, 1080, False)
    app.start()
    #engine = Myengine(1920, 1080)
    #engine.run()

    
