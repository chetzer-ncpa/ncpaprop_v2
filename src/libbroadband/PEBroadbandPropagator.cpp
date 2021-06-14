#include "PEBroadbandPropagator.h"
#include <complex>
#include "binaryreader.h"
#include "util.h"
#include <cstdint>
#include <fstream>
#include <sstream>

NCPA::PEBroadbandPropagator::PEBroadbandPropagator( NCPA::ParameterSet *param ) {
	
	int i, j;
	f_vec = NULL;

	// read in and store parameters
	NFFT 					= param->getInteger( "nfft" );
	waveform_out_file 		= param->getString( "output_waveform_file" );
	tloss_input_file 		= param->getString( "input_tloss_file" );
	source_type 			= param->getString( "source" );
	source_file 			= param->getString( "source_file" );
	f_center 				= param->getFloat( "f_center" );
	max_cel 				= param->getFloat( "max_celerity" );
	single_receiver 		= param->getString("receiver").compare("single") == 0;


	if (single_receiver) {
		Nr = 1;
		r_vec = new double[ 1 ];
		r_vec[ 0 ] = param->getFloat( "range_km" ) * 1000.0;
	} else {
		double rmin, rmax, rstep;
		rmin = param->getFloat( "start_range_km" );
		rmax = param->getFloat( "end_range_km" );
		rstep = param->getFloat( "range_step_km" );
		if (rmax <= rmin) {
			throw std::runtime_error( "end_range_km must be greater than start_range_km" );
		}
		Nr = (int)floor( (rmax-rmin) / rstep ) + 1;
		r_vec = new double[ Nr ];
		for (i = 0; i < Nr; i++) {
			r_vec[ i ] = (rmin + ((double)i) * rstep) * 1000.0;  // expected in meters
		}
	}

	// read output as a function of frequency and range for a given azimuth and height
	// this also sets Nfreq and f_vec
	read_tloss_file( tloss_input_file );

	// check to make sure transfer functions will extend to all ranges
	for (i = 0; i < Nr; i++) {
		for (j = 0; j < Nfreq; j++) {
			if (r_vec[ i ] > TF_[j].r[ TF_[j].nr - 1 ]) {
				std::ostringstream oss;
				oss << "Requested range " << r_vec[i] 
					<< " exceeds maximum range of transfer function calculated for f = "
					<< f_vec[ j ] << " (" << TF_[j].r[ TF_[j].nr - 1 ] << ")" 
					<< std::endl;
				throw std::runtime_error( oss.str() );
			}
		}
	}
	// TL_r  = cmatrix( Nr, Nfreq );
	f_step = f_vec[ 1 ] - f_vec[ 0 ];
	transfer_function = new std::complex<double>[ Nfreq ];
	std::memset( transfer_function, 0, Nfreq * sizeof(std::complex<double>) );

	// if the option --nfft was not passed to the main program then the 
	// NFFT default was 0. Otherwise it has the requested positive value. 
	// Here we make sure that whatever the current value of NFFT is 
	// it's not less than 4*n_freqs
	if (NFFT < (4*Nfreq)) {
		NFFT = 4 * Nfreq;
		std::cout << "Minimum NFFT is set to NFFT = " << NFFT << std::endl;
	}
}

NCPA::PEBroadbandPropagator::~PEBroadbandPropagator() {
	clear_transfer_function_vector_();
	delete [] transfer_function;
}



void NCPA::PEBroadbandPropagator::clear_transfer_function_vector_() {
	std::vector< transfer_function_t >::iterator it;
	for (it = TF_.begin(); it != TF_.end(); ++it) {
		delete [] (*it).r;
		delete [] (*it).TL;
	}
	TF_.clear();
}


// File format:
//
// [header]
// uint32_t n_az
// double az[ 0 ]
// double az[ 1 ]
//   ...
// double az[ n_az-1 ]
// uint32_t n_f
// double  f[ 0 ]
//   ...
// double  f[ n_f-1 ]
//
// [body]
// foreach (az)
//   foreach (freq)
//     uint32_t blocksize
//     double az
//     double freq
//     uint32_t nz
//     double z[ 0 ]
//       ...
//     double z[ nz-1 ]
//     uint32_t nr
//     double r[ 0 ]
//       ...
//     double r[ nr-1 ]
//     double  Re{ TL[ z[0] ][ r[0] ] }
// 	   double  Im{ TL[ z[0] ][ r[0] ] }
// 	   double  Re{ TL[ z[0] ][ r[1] ] }
// 	   double  Im{ TL[ z[0] ][ r[1] ] }
// 	     ...
// 	   double  Re{ TL[ z[0] ][ r[nr-1] ] }
// 	   double  Im{ TL[ z[0] ][ r[nr-1] ] }
// 	   double  Re{ TL[ z[1] ][ r[0] ] }
// 	   double  Im{ TL[ z[1] ][ r[0] ] }
// 	   double  Re{ TL[ z[1] ][ r[1] ] }
// 	   double  Im{ TL[ z[1] ][ r[1] ] }
// 	     ...
void NCPA::PEBroadbandPropagator::read_tloss_file( std::string filename ) {
	
	std::complex< double > J( 0.0, 1.0 );

	//this->reset();
	clear_transfer_function_vector_();

	// read in all values
	std::ifstream ifs( filename, std::ifstream::in | std::ifstream::binary );
	if (!ifs.good()) {
		throw std::runtime_error( "Error opening file to read: " + filename );
	}

	uint32_t uintval;
	double doublebuf[ 2 ];
	std::complex< double > cval;

	// read azimuths
	ifs.read( (char*)(&uintval), sizeof(uint32_t) );
	naz_ = uintval;
	if (az_vec_ != NULL) {
		delete [] az_vec_;
	}
	az_vec_ = new double[ naz_ ];
	ifs.read( (char*)az_vec_, naz_ * sizeof(double) );

	// read frequencies
	ifs.read( (char*)(&uintval), sizeof(uint32_t) );
	Nfreq = uintval;
	if (f_vec != NULL) {
		delete [] f_vec;
	}
	f_vec = new double[ Nfreq ];
	ifs.read( (char*)f_vec, Nfreq * sizeof(double) );

	// read in transfer functions for the specified z value
	size_t az_i, f_i, z_ind, r_i;
	double *z_vec, *rv;
	uint32_t nz, nr;
	for (az_i = 0; az_i < naz_; az_i++) {
		for (f_i = 0; f_i < Nfreq; f_i++) {
			// block size, az, freq.  Don't need it right now
			ifs.seekg( sizeof(uint32_t) + 2*sizeof(double), ifs.cur );

			// read z vector
			ifs.read( (char*)(&nz), sizeof(uint32_t) );
			z_vec = new double[ nz ];
			ifs.read( (char*)z_vec, nz * sizeof(double) );
			z_ind = NCPA::find_closest_index( z_vec, nz, receiver_height );

			// read r vector
			ifs.read( (char*)(&nr), sizeof(uint32_t) );
			rv = new double[ nr ];
			ifs.read( (char*)rv, nr * sizeof(double) );

			// calculate how far into the file to skip
			size_t skipbefore = z_ind * nr * sizeof(double) * 2;
			size_t skipafter  = (nz - z_ind - 1) * nr * sizeof(double) * 2;

			// skip in
			ifs.seekg( skipbefore, ifs.cur );

			// read in the levels
			transfer_function_t tf;
			tf.freq = f_vec[ f_i ];
			tf.nr   = nr;
			tf.r    = new double[ nr ];
			tf.TL   = new std::complex<double>[ nr ];
			for (r_i = 0; r_i < nr; r_i++) {
				tf.r[ r_i ] = rv[ r_i ];
				ifs.read( (char*)doublebuf, 2*sizeof(double) );
				tf.TL[ r_i ] = doublebuf[ 0 ] + J*doublebuf[ 1 ];
			}
			TF_.push_back( tf );

			// skip the rest
			ifs.seekg( skipafter, ifs.cur );

			delete [] z_vec;
			delete [] rv;
		}
	}
	ifs.close();

}

int NCPA::PEBroadbandPropagator::calculate_waveform() {
	
	std::complex<double> *src_vec, *rcv_vec, *tmp_vec;
	size_t r_ind, f_ind;
	double Fmax, tskip, rr, t0, factor;

	// source spectrum
	src_vec = new std::complex<double>[ Nfreq ];
	std::memset( src_vec, 0, Nfreq * sizeof(std::complex<double>) );

	// scratch variable
	tmp_vec = new std::complex<double>[ NFFT ];
	std::memset( tmp_vec, 0, NFFT * sizeof(std::complex<double>) );

	// propagated waveform
	rcv_vec = new std::complex<double>[ NFFT ];
	std::memset( rcv_vec, 0, NFFT * sizeof(std::complex<double>) );

	get_source_spectrum( src_vec, rcv_vec, tmp_vec );

	Fmax = ((double)NFFT) * f_step;

	// @todo unify style here
	std::cout << "--> Propagating pulse from source-to-receivers on grid ..." << std::endl;					      
    printf("----------------------------------------------\n");
    printf("max_celerity     t0             R\n");
    printf("    m/s          sec            km\n");
    printf("----------------------------------------------\n");

    std::ofstream waveout( waveform_out_file );
    tskip = 0.0;
    // DV 20170810 - parameter 'factor' to make it easy to agree with other codes 
    // (e.g. Roger Waxler's modal code)
    factor = 1.0;

    for (r_ind = 0; r_ind < Nr; r_ind++) {
    	rr = r_vec[ r_ind ];
    	t0 = tskip + rr/max_cel;
    	printf("%8.3f     %9.3f      %9.3f\n", max_cel, t0, rr/1000.0);

    	// get transfer function from file-constructed struct
    	calculate_transfer_function( rr );

    	// propagate pulse
    	fft_pulse_prop( t0, rr, src_vec, rcv_vec );

    	for (f_ind = 0; f_ind < NFFT; f_ind++) {
    		waveout.setf( std::ios::fixed, std:: ios::floatfield );
    		waveout.width( 10 );
    		waveout.precision( 3 );
    		waveout << rr / 1000.0 << " ";
    		waveout.width( 12 );
    		waveout.precision( 6 );
    		waveout << 1.0 * f_ind / Fmax + t0 << " ";
    		waveout.setf( std::ios::scientific, std:: ios::floatfield );
    		waveout.width( 15 );
    		waveout << factor * rcv_vec[ f_ind ].real() << std::endl;
    	}
    }

    waveout.close();

    std::cout << "f_step = " << f_step << ", 1/f_step = " << 1.0/f_step << std::endl
    		  << "Time array length = " << NFFT << "; delta_t = " << 1.0/Fmax << std::endl;
    std::cout << "Propagation results saved in file: " << waveform_out_file << std::endl
    		  << "with columns: R (km) | time (s) | pulse(R,t)" << std::endl;

    delete [] src_vec;
    delete [] rcv_vec;
    delete [] tmp_vec;
    return 0;
}

void NCPA::PEBroadbandPropagator::calculate_transfer_function( double range ) {
	
	size_t i;
	int range_ind;

	// clear any existing transfer function
	std::memset( transfer_function, 0, Nfreq*sizeof( std::complex<double> ) );
    	
	// iterate through vector of frequency-dependent transfer functions
	for (i = 0; i < Nfreq; i++) {
		range_ind = NCPA::find_closest_index( TF_[i].r, TF_[i].nr, range );
		transfer_function[ i ] = TF_[i].TL[ range_ind ];
		// std::cout << f_vec[ i ] << ": " << transfer_function[ i ] << std::endl;
	}

	int smooth_space = (int)std::floor( 0.1*Nfreq );
	for(i=Nfreq-smooth_space;i<Nfreq;i++){
		transfer_function[i]=transfer_function[i]*half_hann(Nfreq-smooth_space,Nfreq-1,i); // changed df to 1 as df doesn't make sense here
	}
}

// void NCPA::PEBroadbandPropagator::read_response_spectrum( std::string filename, 
// 	double this_az, double this_z ) {
// }