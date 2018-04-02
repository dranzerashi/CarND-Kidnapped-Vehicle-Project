/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles = 70;

	weights.resize(num_particles);

	particles.resize(num_particles);

	normal_distribution<double> dist_x(x,std[0]);
	normal_distribution<double> dist_y(y,std[1]);
	normal_distribution<double> dist_theta(theta,std[2]);

	for(int i =0; i < num_particles; i++){
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = 1;

	}

	is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	normal_distribution<double> dist_x(0,std_pos[0]);
	normal_distribution<double> dist_y(0,std_pos[1]);
	normal_distribution<double> dist_theta(0,std_pos[2]);


	if(fabs(yaw_rate)<0.0001)
	{
		for(int i =0; i < num_particles; i++){
		
			particles[i].x +=  velocity*delta_t*cos(particles[i].theta) + dist_x(gen);
			particles[i].y +=  velocity*delta_t*sin(particles[i].theta) + dist_y(gen);
			particles[i].theta += dist_theta(gen);
		}
	}
	else{
		for(int i =0; i < num_particles; i++){
		
			particles[i].x +=  velocity/yaw_rate*(sin(particles[i].theta + yaw_rate * delta_t)-sin(particles[i].theta)) + dist_x(gen);
			particles[i].y +=  velocity/yaw_rate*(-cos(particles[i].theta + yaw_rate * delta_t)+cos(particles[i].theta)) + dist_y(gen); + dist_y(gen);
			particles[i].theta += delta_t*yaw_rate  + dist_theta(gen);
		}
	}
	

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for(int i = 0; i < observations.size(); i++){
		double min_dist = dist(observations[i].x,observations[i].y,predicted[0].x,predicted[0].y);
		LandmarkObs nearest_prediction = predicted[0];
		for(int j=1; j<predicted.size(); j++){
			double distance = dist(observations[i].x,observations[i].y,predicted[j].x,predicted[j].y);
			if(distance<min_dist){
				min_dist=distance;
				nearest_prediction = predicted[j];
			}
		}
		observations[i].id = nearest_prediction.id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//Constant 1/2*pi*sigma_x*sigma_y
	const double normalizer = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
	const double two_sigma_x = (2*std_landmark[0]*std_landmark[0]);
	const double two_sigma_y = (2*std_landmark[1]*std_landmark[1]);
	//Create a dictionary that maps id to landmark objects.
	 map <int, LandmarkObs> dict;
	 for(int i=0; i<map_landmarks.landmark_list.size(); i++){
		 LandmarkObs t;
		 t.id  = map_landmarks.landmark_list[i].id_i;
		 t.x  = map_landmarks.landmark_list[i].x_f;
		 t.y  = map_landmarks.landmark_list[i].y_f;
		 dict.insert(pair <int, LandmarkObs> (t.id, t));
	 }


	for(int i = 0; i < num_particles; i++){
		vector<LandmarkObs> predicted;
		for(int j = 0; j < map_landmarks.landmark_list.size(); j++){
			if(dist(particles[i].x,particles[i].y,map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f) <= sensor_range){
				predicted.push_back(dict[map_landmarks.landmark_list[j].id_i]);
			}
		}

		vector<LandmarkObs> observed_map_coord;
		double cos_part_i_theta = cos(particles[i].theta);
		double sin_part_i_theta = sin(particles[i].theta);
		for(int j = 0; j < observations.size(); j++){
			LandmarkObs t;
			t.id = observations[j].id;
			t.x = particles[i].x + (observations[j].x * cos_part_i_theta)-(observations[j].y*sin_part_i_theta);
			t.y = particles[i].y + (observations[j].x * sin_part_i_theta)+(observations[j].y*cos_part_i_theta);
			observed_map_coord.push_back(t);
		}
		
		dataAssociation(predicted, observed_map_coord);
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;

		for(int j = 0; j<observed_map_coord.size(); j++){
			associations.push_back(observed_map_coord[j].id);
			sense_x.push_back(observed_map_coord[j].x);
			sense_y.push_back(observed_map_coord[j].y);
		}
		SetAssociations(particles[i],associations,sense_x,sense_y);


		double w =1.0;
		//cout<<"************ bla"<<endl;
		for(int j=0;j<observed_map_coord.size();j++){
			double mu_x = dict[observed_map_coord[j].id].x;
			double mu_y = dict[observed_map_coord[j].id].y;
			double x = observed_map_coord[j].x;
			double y = observed_map_coord[j].y;
			w*= normalizer*exp(-((pow(x-mu_x,2)/two_sigma_x)+
								(pow(y-mu_y,2)/two_sigma_y)));
		}
		particles[i].weight = w;
		weights[i] = w;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> particle_samples;
	default_random_engine gen;
	uniform_int_distribution<> rand_index(0, num_particles-1);
	uniform_real_distribution<double> rand_num(0.0,1.0);
	double max_w = *max_element(weights.begin(),weights.end());
	double beta = 0.0;
	int index = rand_index(gen);

	for(int i=0; i<num_particles; i++){
		beta += 2 * rand_num(gen) * max_w;
		//cout<<"beta"<<beta<<endl;
		while(beta > weights[index]){
			beta -= weights[index];
			index = (index+1)%num_particles;
		}
		particle_samples.push_back(particles[index]);
	}

	particles = particle_samples;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
