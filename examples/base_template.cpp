
#include "kangEngine.hpp"
#include <iostream>
#include <memory>

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

class MyApp: public App
{
public:
  
    
    void setup() override
    {
        
    }

    void preRender() override
    {   
        
    }

    void render() override
    {
        
    }

    void postRender() override
    {

    }
};

int main(){
    MyApp app;
    app.initialize(SCR_WIDTH, SCR_HEIGHT, false);
    app.start();
    return 0;
}
