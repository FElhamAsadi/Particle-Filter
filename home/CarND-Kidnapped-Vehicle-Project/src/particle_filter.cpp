#include "particle_filter.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
#define EPS 0.0001

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  num_particles = 100;  // TODO: Set the number of particles
  
  // This line creates a normal (Gaussian) distribution for x, y, theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  //Random engine for sampling from normal distribution
  std::default_random_engine gen;
  
  for (int i=0; i<num_particles; i++) { 
   
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;  
    particles.push_back(p);
  }
  
    // Show as initialized; no need for prediction yet
    is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
    
  // This line creates a normal (Gaussian) distribution for x, y, theta
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);   

  //Random engine for sampling from normal distribution
  std::default_random_engine gen;
  
  if (fabs(yaw_rate) <EPS) { 
  	for (int i=0; i<num_particles; i++) { 
    	particles[i].x += velocity* cos(particles[i].theta) * delta_t + dist_x(gen);
    	particles[i].y += velocity* sin(particles[i].theta) * delta_t + dist_y(gen);
        particles[i].theta += dist_theta(gen);
  	}
  }
  else {
    for (int i=0; i<num_particles; i++) { 
    	particles[i].x += velocity/yaw_rate* (sin(particles[i].theta + yaw_rate* delta_t)- sin(particles[i].theta)) + dist_x(gen);
    	particles[i].y += velocity/yaw_rate* (-cos(particles[i].theta + yaw_rate* delta_t) + cos(particles[i].theta)) + dist_y(gen);
    	particles[i].theta +=  yaw_rate * delta_t + dist_theta(gen);
  	}  
  } 
  
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for (unsigned int i=0; i< observations.size(); i++) { 
   	double minDist = dist(observations[i].x, observations[i].y, predicted[0].x, predicted[0].y) ;
      observations[i].id = predicted[0].id;

  	for (unsigned int j=0; j<predicted.size(); j++) { 
      double Dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y) ;
      
      if (Dist< minDist) {
      	observations[i].id = predicted[j].id;
        minDist = Dist;
      }
    }

  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a multi-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   *   NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  for (int i=0; i<num_particles; i++) { 

    // Step 1: Collect the landmarks within the sensor range for each particle, as predictions
    vector<LandmarkObs> InRangeLandmarks;

    for (unsigned int j=0; j<map_landmarks.landmark_list.size(); j++) { 
      double x_land = map_landmarks.landmark_list[j].x_f;
      double y_land = map_landmarks.landmark_list[j].y_f;
      
      double Dist = dist(particles[i].x, particles[i].y, x_land, y_land);
      if (Dist <=sensor_range) { 
      	InRangeLandmarks.push_back({map_landmarks.landmark_list[j].id_i, x_land, y_land} );  //This is in map coordinate
      }
    }
    
    //Step 2: convert the observations from vehicle coordinate to map coordinate,
    // by a rigid transform (rotation & translation), see equation 3.3 from http://planning.cs.uiuc.edu/node99.html
    vector<LandmarkObs> MapCoord_Obs;
    LandmarkObs temp;
    for (unsigned int j=0; j<observations.size(); j++) { 
      temp.x =  particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
      temp.y =  particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
      temp.id = observations[j].id;
      MapCoord_Obs.push_back(temp);
    }
    
    //Step 3: Use dataAssociation(predictions, observations) to find the landmark index for each observation
    dataAssociation(InRangeLandmarks, MapCoord_Obs); 
    
    
    //Step 4: Update the weights of each particle using a multi-variate Gaussian distribution
    double gauss_norm;
    double exponent;
    double weight = 1.0 ;
    gauss_norm = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1] );

    for (unsigned int j=0; j<MapCoord_Obs.size(); j++) {
       
      double mu_x = map_landmarks.landmark_list[MapCoord_Obs[j].id-1].x_f;  
      double mu_y = map_landmarks.landmark_list[MapCoord_Obs[j].id-1].y_f;
                
      exponent = ((MapCoord_Obs[j].x - mu_x)*(MapCoord_Obs[j].x - mu_x) / (2.0 * std_landmark[0]*std_landmark[0]))
               + ((MapCoord_Obs[j].y - mu_y)*(MapCoord_Obs[j].y - mu_y) / (2.0 * std_landmark[1]*std_landmark[1]));    

      weight *= gauss_norm * exp(-exponent); 
    }
  
    particles[i].weight = weight; 
    
    InRangeLandmarks.empty();
    MapCoord_Obs.empty();

  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
   std:: random_device rd;
   std::default_random_engine gen(rd());
  
   vector<double> distribution;
   for (int i=0; i<num_particles; i++) { 
   	 distribution.push_back(particles[i].weight);
   }
  
   // Vector for new particles
   vector<Particle> new_particles (num_particles);

   for (int i = 0; i < num_particles; i++) {
     std::discrete_distribution<> index(distribution.begin(), distribution.end());
     int j = index(gen);
     new_particles[i] = particles[j];
     new_particles[i].weight = 1.0;
   }
  
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}