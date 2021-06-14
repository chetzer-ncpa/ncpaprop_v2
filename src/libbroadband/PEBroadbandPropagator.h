#ifndef NCPAPROP_PEBROADBANDPROPAGATOR_H_INCLUDED
#define NCPAPROP_PEBROADBANDPROPAGATOR_H_INCLUDED

#include "parameterset.h"
#include "BroadbandPropagator.h"
#include <complex>
#include <vector>


typedef struct {
	double freq;
	size_t nr;
	double *r;
	std::complex<double> *TL;
} transfer_function_t;

namespace NCPA {

	class PEBroadbandPropagator : public BroadbandPropagator {

	public:
		PEBroadbandPropagator( ParameterSet *param );
		~PEBroadbandPropagator();

		int calculate_waveform();
		void read_tloss_file( std::string filename );
		void calculate_transfer_function( double range );

	protected:

		//void read_pe_output_file(std::string filename, double az, double range );
		void clear_transfer_function_vector_();
		
		size_t naz_;
		double *az_vec_ = NULL;

		std::vector< transfer_function_t > TF_;
		double receiver_height = 0.0;
		std::string tloss_input_file;
	};

}



#endif
