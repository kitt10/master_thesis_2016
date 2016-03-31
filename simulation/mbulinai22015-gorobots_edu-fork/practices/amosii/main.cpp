/***************************************************************************
 *   Copyright                                                             *
 *    poramate@physik3.gwdg.de                                             *
 *    fhesse@physik3.gwdg.de                                               *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 *                                                                         *
 *   $Log: main.cpp,v $                                                    *
 *                                                                         *
 ***************************************************************************/

// include simulation environment stuff
#include <ode_robots/simulation.h>
// include agent (class for holding a robot, a controller and a wiring)
#include <ode_robots/odeagent.h>
// playground
#include <ode_robots/playground.h>

#include <ode_robots/terrainground.h>

// simple wiring
#include <selforg/one2onewiring.h>
// the robot
#include <ode_robots/amosII.h>

// include the controller
#include "tripod_controller.h"


// joint needed for fixation of the robot in the beginning
#include <ode_robots/joint.h>

// add head file for creating a sphere by Ren ------------
#include <ode_robots/passivesphere.h>
#include <ode_robots/passivebox.h>
#include <selforg/abstractcontroller.h>
#include <ode_robots/color.h>
#include <iostream>

using namespace std;

// fetch all the stuff of lpzrobots into scope
using namespace lpzrobots;

std::vector<lpzrobots::AbstractObstacle*> obst;

int terrain_type_g = 0;
bool terrain_noise = false;
double tn_std_prcnt = 0.0;
bool sim_noise = 0.0;

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

class ThisSim : public lpzrobots::Simulation {
  public:

  ThisSim(){
    addPaletteFile("colors/UrbanExtraColors.gpl");
    addColorAliasFile("colors/UrbanColorSchema.txt");
    // you can replace color mappings in your own file, see colors/UrbanColorSchema.txt
    // addColorAliasFile("myColorSchema.txt");
    setGroundTexture("Images/whiteground.jpg"); // gets its color from the schema
    //setTitle("centered text");
    //setCaption("right aligned text");
  }

  /**
   * starting function (executed once at the beginning of the simulation loop)
   */
  virtual void start(const lpzrobots::OdeHandle& odeHandle,
      const lpzrobots::OsgHandle& osgHandle,
      lpzrobots::GlobalData& global) {
    // set initial camera position
    setCameraHomePos(
        lpzrobots::Pos(-0.0114359, 6.66848, 0.922832),
        lpzrobots::Pos(178.866, -7.43884, 0));

    // set simulation parameters
    global.odeConfig.setParam("controlinterval", 10);
    global.odeConfig.setParam("simstepsize", 0.01);
    global.odeConfig.setParam("noise", sim_noise);
	
	// declare terrain params
	double terrain_roughness;
	double terrain_slip;
	double terrain_hardness;
	double terrain_elasticity;
	double terrain_height;
	Color terrain_color;
	
	int terrain_type = terrain_type_g;
	
	// define individual terrain parameters here
	switch(terrain_type) {
		case 1:	// concrete
			setTitle("concrete");
			terrain_roughness = 10.0;
			terrain_slip = 0.0;
			terrain_hardness = 100.0;
			terrain_elasticity = 0.0;
			terrain_height = 0.0;
			terrain_color = Color(156.0/255.0,159.0/255.0,166.0/255.0);
			break;
		case 2:	// mud
			setTitle("mud");
			terrain_roughness = 0.5;
			terrain_slip = 5.0;
			terrain_hardness = 0.5;
			terrain_elasticity = 0.5;
			terrain_height = 0.02;
			terrain_color = Color(100.0/255.0,100.0/255.0,100.0/255.0);
			break;
		case 3:	// ice
			setTitle("ice");
			terrain_roughness = 0.0;
			terrain_slip = 100.0;
			terrain_hardness = 100.0;
			terrain_elasticity = 0.0;
			terrain_height = 0.0;
			terrain_color = Color(215.0/255.0,227.0/255.0,255.0/255.0);
			break;
		case 4:	// sand
			setTitle("sand");
			terrain_roughness = 1.0;
			terrain_slip = 0.1;
			terrain_hardness = 30.0;
			terrain_elasticity = 0.0;
			terrain_height = 0.02;
			terrain_color = Color(242.0/255.0,238.0/255.0,124.0/255.0);
			break;
		case 5:	// gravel
			setTitle("gravel");
			terrain_roughness = 7.0;
			terrain_slip = 0.1;
			terrain_hardness = 100.0;
			terrain_elasticity = 0.0;
			terrain_height = 0.03;
			terrain_color = Color(115.0/255.0,127.0/255.0,156.0/255.0);
			break;
		case 6:	// grass
			setTitle("grass");
			terrain_roughness = 5.0;
			terrain_slip = 0.0;
			terrain_hardness = 30.0;
			terrain_elasticity = 0.6;
			terrain_height = 0.05;
			terrain_color = Color(35.0/255.0,150.0/255.0,20.0/255.0);
			break;
		case 7:	// swamp
			setTitle("swamp");
			terrain_roughness = 0.0;
			terrain_slip = 5.0;
			terrain_hardness = 0.0;
			terrain_elasticity = 0.0;
			terrain_height = 0.1;
			terrain_color = Color(50.0/255.0,75.0/255.0,50.0/255.0);
			break;
		case 8:	// rock
			setTitle("rock");
			terrain_roughness = 10.0;
			terrain_slip = 0.0;
			terrain_hardness = 100.0;
			terrain_elasticity = 0.0;
			terrain_height = 0.1;
			terrain_color = Color(110.0/255.0,90.0/255.0,60.0/255.0);
			break;
		case 9:	// tiles
			setTitle("tiles");
			terrain_roughness = 5.0;
			terrain_slip = 30.0;
			terrain_hardness = 100.0;
			terrain_elasticity = 0.0;
			terrain_height = 0.0;
			terrain_color = Color(250.0/255.0,200.0/255.0,150.0/255.0);
			break;
		case 10: // snow
			setTitle("snow");
			terrain_roughness = 0.0;
			terrain_slip = 80.0;
			terrain_hardness = 20.0;
			terrain_elasticity = 0.0;
			terrain_height = 0.02;
			terrain_color = Color(255.0/255.0,255.0/255.0,255.0/255.0);
			break;
		case 11: // rubber
			setTitle("rubber");
			terrain_roughness = 8.0;
			terrain_slip = 0.0;
			terrain_hardness = 80.0;
			terrain_elasticity = 2.0;
			terrain_height = 0.0;
			terrain_color = Color(0.0/255.0,0.0/255.0,0.0/255.0);
			break;
		case 12: // carpet
			setTitle("carpet");
			terrain_roughness = 3.0;
			terrain_slip = 0.0;
			terrain_hardness = 40.0;
			terrain_elasticity = 0.3;
			terrain_height = 0.02;
			terrain_color = Color(135.0/255.0,100.0/255.0,150.0/255.0);
			break;
		case 13: // wood
			setTitle("wood");
			terrain_roughness = 6.0;
			terrain_slip = 0.0;
			terrain_hardness = 80.0;
			terrain_elasticity = 0.2;
			terrain_height = 0.02;
			terrain_color = Color(90.0/255.0,65.0/255.0,0.0/255.0);
			break;
		case 14: // plastic
			setTitle("plastic");
			terrain_roughness = 1.0;
			terrain_slip = 2.0;
			terrain_hardness = 60.0;
			terrain_elasticity = 0.5;
			terrain_height = 0.0;
			terrain_color = Color(150.0/255.0,250.0/255.0,190.0/255.0);
			break;
		case 15: // foam
			setTitle("foam");
			terrain_roughness = 5.0;
			terrain_slip = 0.0;
			terrain_hardness = 0.0;
			terrain_elasticity = 0.7;
			terrain_height = 0.05;
			terrain_color = Color(220.0/255.0,230.0/255.0,150.0/255.0);
			break;
		default:
			setTitle("default");
			terrain_roughness = 10.0;
			terrain_slip = 0.0;
			terrain_hardness = 100.0;
			terrain_elasticity = 0.0;
			terrain_height = 0.0;
			terrain_color = Color(255.0/255.0,255.0/255.0,255.0/255.0);
	}
	
	if (terrain_noise) {
		// Adding noise
		terrain_roughness += fRand(-10.0*tn_std_prcnt, 10.0*tn_std_prcnt);
		terrain_slip  += fRand(-10.0*tn_std_prcnt, 10.0*tn_std_prcnt);
		terrain_hardness += fRand(-100.0*tn_std_prcnt, 100.0*tn_std_prcnt);
		terrain_elasticity += fRand(-2.0*tn_std_prcnt, 2.0*tn_std_prcnt);
		terrain_height += fRand(-0.1*tn_std_prcnt, 0.1*tn_std_prcnt);
	}
	
	// limits
	terrain_roughness = max(0.0, terrain_roughness);
	terrain_slip = max(0.0, terrain_slip);
	terrain_hardness = max(0.0, terrain_hardness);
	terrain_elasticity = max(0.0, terrain_elasticity);
	terrain_height = max(0.0, terrain_height);
	
	cout << terrain_roughness << ", " << terrain_slip << ", " << terrain_hardness << ", " << terrain_elasticity << ", " << terrain_height << endl;
	
	//**************Change Material substance*********//
	//Substance roughterrainSubstance(1.0,0.0,/*100.0 friction*/100.0,0.0); //(roughness,slip,hardness,elasticity)
	Substance roughterrainSubstance(terrain_roughness, terrain_slip, terrain_hardness, terrain_elasticity); //(roughness(0:smooth/1:rough), slip(0:friction/100:sliding), hardness(0:soft/100:hard), elasticity(0:hard/1:elastic))
	OdeHandle oodeHandle = odeHandle;
	oodeHandle.substance = roughterrainSubstance;
	//**************Change Material substance*********//

	TerrainGround* terrainground =
			new TerrainGround(oodeHandle, osgHandle.changeColor(terrain_color),
								"rough1.ppm","", /*AREA-SIZE-X->*/25, /*AREA-SIZE-Y->*/25, /*HEIGHT->*/terrain_height);
	terrainground->setPose(osg::Matrix::translate(0, 0, 0.005));
	global.obstacles.push_back(terrainground);

	
    /* ########################## Add amosII robot ########################## */
    lpzrobots::AmosIIConf myAmosIIConf = lpzrobots::AmosII::getDefaultConf(1.0 /*_scale*/,1 /*_useShoulder*/,1 /*_useFoot*/,1 /*_useBack*/);
    myAmosIIConf.rubberFeet = true;
    lpzrobots::OdeHandle rodeHandle = odeHandle;
    rodeHandle.substance = lpzrobots::Substance(3.0, 0.0, 50.0, 0.8);
    
    amos = new lpzrobots::AmosII(
        rodeHandle,
        osgHandle.changeColor(lpzrobots::Color(1, 1, 1)),
        myAmosIIConf, "AmosII");

    // define the usage of the individual legs
    amos->setLegPosUsage(amos->L0, amos->LEG);
    amos->setLegPosUsage(amos->L1, amos->LEG);
    amos->setLegPosUsage(amos->L2, amos->LEG);
    amos->setLegPosUsage(amos->R0, amos->LEG);
    amos->setLegPosUsage(amos->R1, amos->LEG);
    amos->setLegPosUsage(amos->R2, amos->LEG);

    // put amos on a random position in the map
    double init_pos_x = fRand(0.0, 5.0);
    double init_pos_y = fRand(0.0, 5.0);
    cout << init_pos_x << "," << init_pos_y;
    amos->place(osg::Matrix::translate(init_pos_x, init_pos_y, 0.1));

    controller = new EmptyController();
    // create wiring
    One2OneWiring* wiring = new One2OneWiring(new ColorUniformNoise());

    // create agent and init it with controller, robot and wiring
    lpzrobots::OdeAgent* agent = new lpzrobots::OdeAgent(global);
    agent->init(controller, amos, wiring);
	
	
    // create a fixed joint to hold the robot in the air at the beginning
    robotfixator = new lpzrobots::FixedJoint(
        amos->getMainPrimitive(),
        global.environment);
    robotfixator->init(odeHandle, osgHandle, false);
    

    // inform global variable over everything that happened:
    global.configs.push_back(amos);
    global.agents.push_back(agent);
    global.configs.push_back(controller);
   	
   	// remove robotfixator
    if (robotfixator) {
      delete robotfixator;
      robotfixator = NULL;
    }
  }

  /**
   * add own key handling stuff here, just insert some case values
   */
  virtual bool command(const lpzrobots::OdeHandle&,
      const lpzrobots::OsgHandle&,
      lpzrobots::GlobalData& globalData,
      int key,
      bool down)
  {
    if (down) { // only when key is pressed, not when released
      switch (char(key)) {
        case 'x':
          if (robotfixator) {
            std::cout << "dropping robot" << std::endl;
            delete robotfixator;
            robotfixator = NULL;
          }
          break;
        default:
          return false;
          break;
      }
    }
    return false;
  }
  protected:
  lpzrobots::Joint* robotfixator;
  AbstractController* controller;
  lpzrobots::AmosII* amos;
};


/* 			$ ./start terrain_type
	e.g. 	$ ./start 4 
*/
int main(int argc, char **argv)
{
  ThisSim sim;
  sim.setGroundTexture("Images/greenground.rgb");
  if (argc > 1) {
  	terrain_type_g = atoi(argv[1]);
  }
  if (argc > 2) {
  	tn_std_prcnt = atof(argv[2]);
  	terrain_noise = true;
  	cout << "terrain_noise on" << endl;
  }
  if (argc > 3) {
  	sim_noise = atof(argv[3]);
  }
  
  return sim.run(argc, argv) ? 0 : 1;
}

