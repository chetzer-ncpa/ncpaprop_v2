#include "EPadeSolver.h"
//#include "epade_pe_parameters.h"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <string>
#include <vector>
#include <cfloat>
#include <fstream>
#include <stdexcept>
#include <cstdint>

#include "petscksp.h"

#include "Atmosphere1D.h"
#include "Atmosphere2D.h"
#include "ToyAtmosphere1D.h"
#include "StratifiedAtmosphere2D.h"
#include "ProfileSeriesAtmosphere2D.h"
#include "units.h"
#include "util.h"
#include "parameterset.h"


#ifndef PI
#define PI 3.14159
#endif

#define RHO_B 301.0

int outputQ( std::string filename, Mat *Q, PetscInt NZ ) {
	PetscInt ncols;
	const PetscInt *inds;
	const PetscScalar *vals;
	PetscErrorCode ierr;
	std::ofstream out( filename );
	for (PetscInt i = 0; i < NZ; i++) {
		ierr = MatGetRow( *Q, i, &ncols, &inds, &vals );
		for (PetscInt j = 0; j < ncols; j++) {
			out << i << " " << inds[j] << " " << vals[j].real() << " " << vals[j].imag()
				<< std::endl;
		}
		ierr = MatRestoreRow( *Q, i, &ncols, &inds, &vals );
	}
	out.close();
	return 1;
}

void outputVec( Vec &v, double *z, int n, std::string filename ) {
	PetscScalar *array;
	std::ofstream out( filename );
	VecGetArray(v,&array);
	for (int i = 0; i < n; i++) {
		out << z[i] << "  " << array[i].real() << "  " << array[i].imag() << std::endl;
	}
	VecRestoreArray(v,&array);
	out.close();
}

void outputArray( std::complex<double> *array, double *z, int n, std::string filename ) {
	//PetscScalar *array;
	std::ofstream out( filename );
	//VecGetArray(v,&array);
	for (int i = 0; i < n; i++) {
		out << z[i] << "  " << array[i].real() << "  " << array[i].imag() << std::endl;
	}
	//VecRestoreArray(v,&array);
	out.close();
}

void printVector( std::string title, std::vector< PetscScalar > &v ) {
	std::cout << std::endl << title << std::endl;
	for (std::vector<PetscScalar>::const_iterator i = v.cbegin(); i != v.cend(); ++i) {
		std::cout << (*i).real() << ((*i).imag() < 0 ? " - " : " + " ) 
				  << std::fabs( (*i).imag() ) << "*i" << ", ";
	}
	std::cout << std::endl << std::endl;
}

int invertMat( Mat *M, PetscInt NZ, Mat *Mi ) {
	PetscErrorCode ierr;
	Mat S, Cf;
	PetscInt i;
	PetscScalar temp = 1;

	ierr = MatCreateSeqDense( PETSC_COMM_SELF, NZ, NZ, PETSC_NULL, &S );CHKERRQ(ierr);
	ierr = MatSetFromOptions( S );CHKERRQ(ierr);
	for (i = 0; i < NZ; i++) {
		MatSetValues( S, 1, &i, 1, &i, &temp, INSERT_VALUES );CHKERRQ(ierr);
	}
	ierr = MatAssemblyBegin( S, MAT_FINAL_ASSEMBLY );CHKERRQ(ierr);
	ierr = MatAssemblyEnd( S, MAT_FINAL_ASSEMBLY );CHKERRQ(ierr);

	ierr = MatDuplicate( *M, MAT_COPY_VALUES, &Cf );CHKERRQ(ierr);
	ierr = MatAssemblyBegin( Cf, MAT_FINAL_ASSEMBLY );CHKERRQ(ierr);
	ierr = MatAssemblyEnd( Cf, MAT_FINAL_ASSEMBLY );CHKERRQ(ierr);

	ierr = MatLUFactor( Cf, PETSC_NULL, PETSC_NULL, PETSC_NULL );CHKERRQ(ierr);
	ierr = MatMatSolve( Cf, S, *Mi );CHKERRQ(ierr);

	ierr = MatDestroy( &S );CHKERRQ(ierr);
	ierr = MatDestroy( &Cf );CHKERRQ(ierr);
	return 1;
}


NCPA::EPadeSolver::EPadeSolver( NCPA::ParameterSet *param ) {

	c_underground		= 5000.0;

	// obtain the parameter values from the user's options
	// @todo add units to input scalar quantities
	r_max	 			= param->getFloat( "maxrange_km" ) * 1000.0;
  	z_max	 			= param->getFloat( "maxheight_km" ) * 1000.0;      // @todo fix elsewhere that m is required
  	zs			 		= param->getFloat( "sourceheight_km" ) * 1000.0;
  	zr 					= param->getFloat( "receiverheight_km" ) * 1000.0;
  	NR_requested 		= param->getInteger( "Nrng_steps" );
  	freq 				= param->getFloat( "freq" );
	npade 				= param->getInteger( "npade" );
	starter 			= param->getString( "starter" );
	attnfile 			= param->getString( "attnfile" );

	// flags
	lossless 			= param->wasFound( "lossless" );
	use_atm_1d			= param->wasFound( "atmosfile" );
	use_atm_2d			= param->wasFound( "atmosfile2d" );
	//use_atm_toy			= param->wasFound( "toy" );
	top_layer			= !(param->wasFound( "disable_top_layer" ));
	use_topo			= param->wasFound( "topo" );
	write2d 			= param->wasFound( "write_2d_tloss" );
	multiprop 			= param->wasFound( "multiprop" );
	broadband = false;
	user_ground_impedence 	= std::complex<double>( 0.0, 0.0 );

	// Handle differences based on single vs multiprop
	double min_az, max_az, step_az;
	if (multiprop) {
		if (use_atm_2d) {
			std::cerr << "Range-dependent 2-D atmosphere incompatible with multiple azimuth propagation"
					  << std::endl;
			exit(0);
		}
		if (write2d) {
			std::cout << "Multi-azimuth propagation requested, disabling 2-D output" << std::endl;
			write2d = false;
		}
		if (use_topo) {
			std::cout << "Multi-azimuth propagation requested, disabling topography flag" << std::endl;
			use_topo = false;
		}
		min_az 			= param->getFloat( "azimuth_start" );
		max_az 			= param->getFloat( "azimuth_end" );
		step_az 		= param->getFloat( "azimuth_step" );
		NAz 			= (int) ((max_az - min_az)/step_az) + 1;

		// initialize output file
		std::cout << "Initializing file " << NCPAPROP_EPADE_PE_FILENAME_MULTIPROP << std::endl;
		std::ofstream truncator( NCPAPROP_EPADE_PE_FILENAME_MULTIPROP, 
			std::ofstream::out | std::ofstream::trunc );
		truncator.close();

	} else {
		NAz 			= 1;
		min_az 			= param->getFloat( "azimuth" );
		max_az 			= min_az;
		step_az 		= 0;
	}
	azi = new double[ NAz ];
	memset( azi, 0, NAz * sizeof( double ) );
	for (int i = 0; i < NAz; i++) {
		azi[ i ] = min_az + i * step_az;
	}

  	if (broadband) {
  		if (write2d) {
			std::cout << "Broadband propagation requested, disabling 2-D output" << std::endl;
			write2d = false;
		}
		if (use_topo) {
			std::cout << "Broadband propagation requested, disabling topography flag" << std::endl;
			use_topo = false;
		}
		double f_min, f_step, f_max;
      	f_min = param->getFloat( "f_min" );
      	f_step = param->getFloat( "f_step" );
      	f_max = param->getFloat( "f_max" );

      	// sanity checks
      	if (f_min >= f_max) {
      		throw std::runtime_error( "f_min must be less than f_max!" );
      	}

      	NF = (double)(std::floor((f_max - f_min) / f_step)) + 1;
      	f = new double[ NF ];
      	std::memset( f, 0, NF * sizeof( double ) );
      	for (int fi = 0; fi < NF; fi++) {
      		f[ fi ] = f_min + ((double)fi) * f_step;
      	}
	} else {
		freq = param->getFloat( "freq" );
		f = new double[ 1 ];
		f[ 0 ] = freq;
		NF = 1;
	}

	//NCPA::Atmosphere1D *atm_profile_1d;
	if (use_atm_1d) {
		atm_profile_2d = new NCPA::StratifiedAtmosphere2D( param->getString( "atmosfile" ), param->getString("atmosheaderfile") );
	// } else if (use_atm_toy) {
	// 	NCPA::Atmosphere1D *tempatm = new NCPA::ToyAtmosphere1D();
	// 	atm_profile_2d = new NCPA::StratifiedAtmosphere2D( tempatm );
	// 	delete tempatm;
	} else if (use_atm_2d) {
		atm_profile_2d = new NCPA::ProfileSeriesAtmosphere2D( param->getString( "atmosfile2d" ), param->getString( "atmosheaderfile" ) );
		atm_profile_2d->convert_range_units( NCPA::Units::fromString( "m" ) );
		if (r_max > atm_profile_2d->get_maximum_valid_range() ) {
			atm_profile_2d->set_maximum_valid_range( r_max );
		}
	} else {
		std::cerr << "Unknown atmosphere option selected" << std::endl;
		exit(0);
	}

	z_min = atm_profile_2d->get_minimum_altitude( 0.0 );
	z_ground = z_min;
	if (param->wasFound("groundheight_km")) {
		z_ground = param->getFloat( "groundheight_km" ) * 1000.0;
		z_ground_specified = true;
		
		std::cout << "Overriding profile Z0 value with command-line value " << z_ground 
		     << " m" << std::endl;
		atm_profile_2d->remove_property("Z0");
		atm_profile_2d->add_property( "Z0", z_ground, NCPA::Units::fromString("m") );
	} else {
		if (!(atm_profile_2d->contains_scalar(0,"Z0"))) {
			atm_profile_2d->add_property("Z0",z_ground,atm_profile_2d->get_altitude_units(0.0));
		}
	}

	// set units
	atm_profile_2d->convert_altitude_units( Units::fromString( "m" ) );
	atm_profile_2d->convert_property_units( "Z0", Units::fromString( "m" ) );
	atm_profile_2d->convert_property_units( "U", Units::fromString( "m/s" ) );
	atm_profile_2d->convert_property_units( "V", Units::fromString( "m/s" ) );
	atm_profile_2d->convert_property_units( "T", Units::fromString( "K" ) );
	atm_profile_2d->convert_property_units( "P", Units::fromString( "Pa" ) );
	atm_profile_2d->convert_property_units( "RHO", Units::fromString( "kg/m3" ) );
	z_ground = atm_profile_2d->get( 0.0, "Z0" );

	// calculate derived quantities
	for (std::vector< NCPA::Atmosphere1D * >::iterator it = atm_profile_2d->first_profile();
		 it != atm_profile_2d->last_profile(); ++it) {
		if ( (*it)->contains_vector( "C0" ) ) {
			(*it)->convert_property_units( "C0", Units::fromString( "m/s" ) );
			(*it)->copy_vector_property( "C0", "_C0_" );
		} else {
			if ( (*it)->contains_vector("P") && (*it)->contains_vector("RHO") ) {
				(*it)->calculate_sound_speed_from_pressure_and_density( "_C0_", "P", "RHO", 
					Units::fromString( "m/s" ) );
			} else {
				(*it)->calculate_sound_speed_from_temperature( "_C0_", "T",
					Units::fromString( "m/s" ) );
			}
		}
	}
	atm_profile_2d->calculate_wind_speed( "_WS_", "U", "V" );
	atm_profile_2d->calculate_wind_direction( "_WD_", "U", "V" );
	if ( attnfile.size() > 0 ) {
		atm_profile_2d->read_attenuation_from_file( "_ALPHA_", param->getString( "attnfile" ) );
	}

	// calculate/check z resolution
	dz = 				param->getFloat( "dz_m" );
	double c0 = atm_profile_2d->get( 0.0, "_C0_", z_ground );
	double lambda0 = c0 / freq;
  	if (dz <= 0.0) {
  		dz = lambda0 / 20.0;
  		double nearestpow10 = std::pow( 10.0, std::floor( std::log10( dz ) ) );
  		double factor = std::floor( dz / nearestpow10 );
  		dz = nearestpow10 * factor;
  		std::cout << "Setting dz to " << dz << " m" << std::endl;
  	}
  	if (dz > (c0 / freq / 10.0) ) {
  		std::cerr << "Altitude resolution is too coarse.  Must be <= " << lambda0 / 10.0 << " meters." 
  			<< std::endl;
  		exit(0);
  	}

  	// calculate ground impedence
  	if (param->wasFound( "ground_impedence_real" ) || param->wasFound( "ground_impedence_imag" ) ) {
  		user_ground_impedence.real( param->getFloat( "ground_impedence_real" ) );
  		user_ground_impedence.imag( param->getFloat( "ground_impedence_imag" ) );
  		user_ground_impedence_found = true;
  	}
}

NCPA::EPadeSolver::~EPadeSolver() {
	delete [] z;
	delete [] z_abs;
	delete [] azi;
	delete atm_profile_2d;
}

int NCPA::EPadeSolver::solve() {
	if (use_topo) {
		return solve_with_topography();
	} else {
		return solve_without_topography();
	}
}

int NCPA::EPadeSolver::solve_without_topography() {
	int i;
	std::complex<double> I( 0.0, 1.0 );
	PetscErrorCode ierr;
	PetscInt *indices;
	PetscScalar hank, *contents;
	Mat B, C;   // , q;
	Mat *qpowers = PETSC_NULL, *qpowers_starter = PETSC_NULL;
	Vec psi_o, Bpsi_o; //, psi_temp;
	KSP ksp;
	// PC pc;

	// set up z grid for flat ground.  When we add terrain we will need to move this inside
	// the range loop
	//int profile_index = atm_profile_2d->get_profile_index( 0.0 );
	int profile_index;
	double minlimit, maxlimit;
	atm_profile_2d->get_maximum_altitude_limits( minlimit, maxlimit );
	z_max = NCPA::min( z_max, minlimit );    // lowest valid top value
	int ground_index = 0;
	std::complex<double> ground_impedence_factor( 0.0, 0.0 );

	// truncate multiprop file if needed
	if (multiprop) {
		std::ofstream ofs( NCPAPROP_EPADE_PE_FILENAME_2D, std::ofstream::out | std::ofstream::trunc );
		ofs.close();
	}

	// truncate 1-D if necessary
	if (broadband) {
		std::ofstream ofs( NCPAPROP_EPADE_PE_FILENAME_1D, std::ofstream::out | std::ofstream::trunc );
		ofs.close();
	}

	/* @todo move this into constructor as much as possible */
	// if (use_topo) {
	// 	z_bottom = -5000.0;    // make this eventually depend on frequency
	// 	z_bottom -= fmod( z_bottom, dz );
	// 	z_ground = atm_profile_2d->get( 0.0, "Z0" );
	// 	NZ = ((int)std::floor((z_max - z_bottom) / dz)) + 1;
	// 	z = new double[ NZ ];
	// 	z_abs = new double[ NZ ];
	// 	std::memset( z, 0, NZ * sizeof( double ) );
	// 	std::memset( z_abs, 0, NZ * sizeof( double ) );
	// 	indices = new PetscInt[ NZ ];
	// 	std::memset( indices, 0, NZ*sizeof(PetscInt) );
	// 	for (i = 0; i < NZ; i++) {
	// 		z[ i ] = ((double)i) * dz + z_bottom;
	// 		z_abs[ i ] = z[ i ];
	// 		indices[ i ] = i;
	// 	}
	// 	zs = NCPA::max( zs, z_ground );
	// 	ground_index = NCPA::find_closest_index( z, NZ, z_ground );
	// 	if ( z[ ground_index ] < z_ground ) {
	// 		ground_index++;
	// 	}
	// 	// if (z[ ground_index ] > z_ground && ground_index > 0) {
	// 	// 	ground_index--;
	// 	// }

	// } else {
		atm_profile_2d->get_minimum_altitude_limits( minlimit, z_min );
		//z_min = atm_profile_2d->get_highest_minimum_altitude();
		if ( (!z_ground_specified) && atm_profile_2d->contains_scalar( 0.0, "Z0" )) {
			z_ground = atm_profile_2d->get( 0.0, "Z0" );
		}
		if (z_ground < z_min) {
			std::cerr << "Supplied ground height is outside of atmospheric specification." << std::endl;
			exit(0);
		}
	  	z_bottom = z_min;
		// fill and convert to SI units
		//double dz       = (z_max - z_ground)/(NZ - 1);	// the z-grid spacing
		NZ = ((int)std::floor((z_max - z_ground) / dz)) + 1;
		z = new double[ NZ ];
		z_abs = new double[ NZ ];
		std::memset( z, 0, NZ * sizeof( double ) );
		std::memset( z_abs, 0, NZ * sizeof( double ) );
		indices = new PetscInt[ NZ ];
		std::memset( indices, 0, NZ*sizeof(PetscInt) );
		for (i = 0; i < NZ; i++) {
			z[ i ]     = ((double)i) * dz;
			z_abs[ i ] = z[ i ] + z_ground;
			indices[ i ] = i;
		}
		zs = NCPA::max( zs-z_ground+dz, dz );
	// }
	// tl = NCPA::cmatrix( NZ, NR-1 );
	
	/*
	int plotz = 10;
	for (i = ((int)fmod((double)ground_index,(double)plotz)); i < NZ; i += plotz) {
		zt.push_back( z_abs[ i ] );
		zti.push_back( i );
	}
	nzplot = zt.size();
	*/

	// constants for now
	//double omega = 2.0 * PI * freq;
	//double dr = r[1] - r[0];
	//double h = z[1] - z[0];
	double h = dz;
	double h2 = h * h;
	double dr;
	//ground_impedence = std::complex<double>( 1.0 / h2, 0.0 );

	// set up for source atmosphere
	double k0 = 0.0, c0 = 0.0;
	double *c = new double[ NZ ];
	double *a_t = new double[ NZ ];
	std::complex<double> *k = new std::complex<double>[ NZ ];
	std::complex<double> *n = new std::complex<double>[ NZ ];

	// write broadband header for testing
	if (broadband) {
		write_broadband_header( NCPAPROP_EPADE_PE_FILENAME_BROADBAND, azi, NAz, f, NF, 1.0e8 );
	}

	// freq and calc_az hold the current values of azimuth and frequency, respectively
	// these are used in the output routines, so make sure they get set correctly
	// whenever you change frequencies and azimuths
	for (int azind = 0; azind < NAz; azind++) {
		calc_az = azi[ azind ];
		std::cout << "Infrasound PE code at f = " << freq << " Hz, azi = " 
			<< calc_az << " deg" << std::endl;

		profile_index = -1;
		atm_profile_2d->calculate_wind_component( "_WC_", "_WS_", "_WD_", calc_az );
		atm_profile_2d->calculate_effective_sound_speed( "_CEFF_", "_C0_", "_WC_" );

		for (int freqind = 0; freqind < NF; freqind++) {

			freq = f[ freqind ];
			if (attnfile.length() == 0) {
				atm_profile_2d->calculate_attenuation( "_ALPHA_", "T", "P", "RHO", freq );
			}

			if (NR_requested == 0) {
		  		dr = 340.0 / freq;
				NR = (int)ceil( r_max / dr );
		  	} else {
		  		NR = NR_requested;
		  		dr = r_max / NR;
		  	}
		  	r = new double[ NR ];
		  	std::memset( r, 0, NR * sizeof(double) );
		  	zgi_r = new int[ NR ];
		  	std::memset( zgi_r, 0, NR * sizeof( int ) );
		  	int i;
		  	for (i = 0; i < NR; i++) {
		  		r[ i ] = ((double)(i+1)) * dr;
		  	}		
			tl = NCPA::cmatrix( NZ, NR-1 );

			// calculate ground impedence (script A in notes in eq. 12)
			double rho0 = atm_profile_2d->get( 0.0, "RHO", z_ground );
			double lambBC = atm_profile_2d->get_first_derivative( 0.0, "RHO", z_ground ) / (2.0 * rho0);
			//lambBC = 0.0;
			if (user_ground_impedence_found) {
				ground_impedence_factor = I * 2.0 * PI * freq * rho0 / user_ground_impedence + lambBC;
				std::cout << "Using user ground impedence of " << user_ground_impedence << std::endl;
				//		<< " results in calculated A factor of " << ground_impedence_factor << std::endl;
			} else {
				ground_impedence_factor.real( lambBC );
				ground_impedence_factor.imag( 0.0 );
				std::cout << "Using default rigid ground with Lamb BC" << std::endl;
				//: A factor = " << ground_impedence_factor << std::endl;
			}

			//std::cout << "Using atmosphere index " << profile_index << std::endl;
			calculate_atmosphere_parameters( atm_profile_2d, NZ, z, 0.0, z_ground, lossless, 
				top_layer, freq, use_topo, k0, c0, c, a_t, k, n );


			// calculate q matrices
			Mat q;
			//qpowers = new Mat[ npade+1 ];
			//qpowers_starter = new Mat[ npade+1 ];
			// make_q_powers( NZ, z, k0, h2, ground_impedence_factor, n, npade+1, 0, qpowers );
			// PetscBool symm;
			// ierr = MatIsSymmetric( qpowers[0], 1e-8, &symm );
			// std::cout << "Matrix q[0] " << (symm == PETSC_TRUE ? "is" : "is not") << " symmetric" << std::endl;
			// exit(0);

			build_operator_matrix_without_topography( NZ, z, k0, h2, ground_impedence_factor, 
				n, npade+1, 0, &q );
			create_polymatrix_vector( npade+1, &q, &qpowers );
			ierr = MatDestroy( &q );CHKERRQ(ierr);

			if (starter == "self") {
				Mat q_starter;
				// qpowers_starter = new Mat[ npade+1 ];
				// make_q_powers( NZ, z, k0, h2, ground_impedence_factor, n, npade+1, ground_index, qpowers_starter );
				build_operator_matrix_without_topography( NZ, z, k0, h2, ground_impedence_factor, 
					n, npade+1, 0, &q_starter );
				create_polymatrix_vector( npade+1, &q_starter, &qpowers_starter );
				// Output Q[0] for testing of the starter
				// PetscInt ncols;
				// const PetscScalar *colvals;
				// for (PetscInt rownum = 0; rownum < NZ; rownum++) {
				// 	ierr = MatGetRow( qpowers_starter[ 0 ], rownum, &ncols, NULL, &colvals );CHKERRQ(ierr);
				// 	std::cout << z[ rownum ] << " " << ncols << " ";
				// 	for (PetscInt colnum = 0; colnum < ncols; colnum++) {
				// 		std::cout << colvals[colnum].real() << " " << colvals[colnum].imag()
				// 				  << " ";
				// 	}
				// 	std::cout << std::endl;
				// 	ierr = MatRestoreRow( qpowers_starter[ 0 ], rownum, &ncols, NULL, &colvals );CHKERRQ(ierr);
				// }
				// PetscBool symm;
				// ierr = MatIsSymmetric( qpowers_starter[0], 1e-8, &symm );
				// std::cout << "Matrix q[0] " << (symm == PETSC_TRUE ? "is" : "is not") << " symmetric" << std::endl;
				// exit(0);
				get_starter_self_revised( NZ, z, zs, r[ 0 ], z_ground, k0, qpowers_starter, 
					npade, &psi_o );
				ierr = MatDestroy( &q_starter );CHKERRQ(ierr);
			} else if (starter == "gaussian") {
				qpowers_starter = qpowers;
				get_starter_gaussian( NZ, z, zs, k0, ground_index, &psi_o );
			} else {
				std::cerr << "Unrecognized starter type: " << starter << std::endl;
				exit(0);
			}

			std::cout << "Outputting starter..." << std::endl;
			outputVec( psi_o, z, NZ, "starter_new.dat" );

			std::cout << "Finding ePade coefficients..." << std::endl;
			std::vector< std::complex<double> > P, Q;
			// epade( npade, k0, dr, &P, &Q );
			// make_B_and_C_matrices( qpowers_starter, npade, NZ, P, Q, &B, &C );
			std::vector< PetscScalar > taylor = taylor_exp_id_sqrt_1pQ_m1( 2*npade, k0*dr );
			calculate_pade_coefficients( &taylor, npade, npade+1, &P, &Q );
			generate_polymatrices( qpowers_starter, npade, NZ, P, Q, &B, &C );

			std::cout << "Marching out field..." << std::endl;
			ierr = VecDuplicate( psi_o, &Bpsi_o );CHKERRQ(ierr);
			contents = new PetscScalar[ NZ ];

			ierr = KSPCreate( PETSC_COMM_SELF, &ksp );CHKERRQ(ierr);
			ierr = KSPSetOperators( ksp, C, C );CHKERRQ(ierr);
			ierr = KSPSetFromOptions( ksp );CHKERRQ(ierr);
			for (PetscInt ir = 0; ir < (NR-1); ir++) {

				double rr = r[ ir ];
				// check for atmosphere change
				if (((int)(atm_profile_2d->get_profile_index( rr ))) != profile_index) {
				
					profile_index = atm_profile_2d->get_profile_index( rr );
					calculate_atmosphere_parameters( atm_profile_2d, NZ, z, rr, z_ground, lossless, top_layer, freq, 
						use_topo, k0, c0, c, a_t, k, n );
					// std::ostringstream oss;
					// oss << "k." << ir << ".dat";
					// outputArray( k, z, NZ, oss.str() );
					delete_polymatrix_vector( npade+1, &qpowers );
					// for (i = 0; i < npade+1; i++) {
					// 	ierr = MatDestroy( qpowers + i );CHKERRQ(ierr);
					// }
					build_operator_matrix_without_topography( NZ, z, k0, h2, 
						ground_impedence_factor, n, npade+1, 0, &q );
					create_polymatrix_vector( npade+1, &q, &qpowers );
					ierr = MatDestroy( &q );

					// epade( npade, k0, dr, &P, &Q );
					taylor.clear();
					taylor = taylor_exp_id_sqrt_1pQ_m1( 2*npade, k0*dr );
					calculate_pade_coefficients( &taylor, npade, npade+1, &P, &Q );
					ierr = MatZeroEntries( B );CHKERRQ(ierr);
					ierr = MatZeroEntries( C );CHKERRQ(ierr);
					// make_B_and_C_matrices( qpowers, npade, NZ, P, Q, &B, &C );
					generate_polymatrices( qpowers, npade, NZ, P, Q, &B, &C );
					std::cout << "Switching to atmosphere index " << profile_index 
						<< " at range = " << rr/1000.0 << " km" << std::endl;
				}

				hank = sqrt( 2.0 / ( PI * k0 * rr ) ) * exp( I * ( k0 * rr - PI/4.0 ) );
				ierr = VecGetValues( psi_o, NZ, indices, contents );
				for (i = 0; i < NZ; i++) {
					tl[ i ][ ir ] = contents[ i ] * hank;
				}

				// if (use_topo) {
				// 	double z0g = atm_profile_2d->get( rr, "Z0" );
				// 	z0g = NCPA::max( z0g, zr );
				// 	zgi_r[ ir ] = NCPA::find_closest_index( z, NZ, z0g );
				// 	if ( z[ zgi_r[ ir ] ] < z0g ) {
				// 		zgi_r[ ir ]++;
				// 	}
				// } else {
				zgi_r[ ir ] = 0.0;
				// }


				if ( fmod( rr, 1.0e5 ) < dr) {
					std::cout << " -> Range " << rr/1000.0 << " km" << std::endl;
				}

				ierr = MatMult( B, psi_o, Bpsi_o );CHKERRQ(ierr);
				ierr = KSPSetOperators( ksp, C, C );CHKERRQ(ierr);  // may not be necessary
				ierr = KSPSolve( ksp, Bpsi_o, psi_o );CHKERRQ(ierr);
				// std::ostringstream oss;
				// oss << "Bpsi_o." << ir << ".dat";
				// outputVec( Bpsi_o, z, NZ, oss.str() );

				// std::ostringstream oss;
				// oss << "step" << ir << ".dat";
				// std::ofstream compfile( oss.str(), std::ios_base::out );
				// compfile << "ir = " << ir << std::endl;
				// compfile << "rr = " << rr << std::endl;
				// compfile << "z_ground = " << z_ground << std::endl;
				// compfile << "k0 = " << k0 << std::endl;
				// compfile << "c0 = " << c0 << std::endl;
				// compfile << "h2 = " << h2 << std::endl;
				// compfile << "ground_impedence_factor = " << ground_impedence_factor << std::endl;
				// compfile << "hank = " << hank << std::endl;
				// compfile << "P = { ";
				// for (std::vector<PetscScalar>::const_iterator ci = P.cbegin(); 
				// 	 ci != P.cend(); ++ci) {
				// 	compfile << *ci << ", ";
				// }
				// compfile << "}" << std::endl;
				// compfile << "Q = { ";
				// for (std::vector<PetscScalar>::const_iterator ci = Q.cbegin(); 
				// 	 ci != Q.cend(); ++ci) {
				// 	compfile << *ci << ", ";
				// }
				// compfile << "}" << std::endl;
				// compfile.close();
			}
			std::cout << "Stopped at range " << r[ NR-1 ]/1000.0 << " km" << std::endl;

			

			if (multiprop) {
				if (write1d) {
					std::cout << "Writing 1-D output to " << NCPAPROP_EPADE_PE_FILENAME_MULTIPROP << std::endl;
					output1DTL( NCPAPROP_EPADE_PE_FILENAME_MULTIPROP, true );
				}
			} else { 
				if (write1d) {
					std::cout << "Writing 1-D output to " << NCPAPROP_EPADE_PE_FILENAME_1D << std::endl;
					output1DTL( NCPAPROP_EPADE_PE_FILENAME_1D, broadband );
				}
				if (write2d) {
					std::cout << "Writing 2-D output to " << NCPAPROP_EPADE_PE_FILENAME_2D << std::endl;
					output2DTL( NCPAPROP_EPADE_PE_FILENAME_2D );
				}
			}

			// write broadband body for testing
			if (broadband) {
				write_broadband_results( NCPAPROP_EPADE_PE_FILENAME_BROADBAND, calc_az, freq, r, NR, z_abs, NZ, tl, 1.0e8 );
			}
			
			std::cout << std::endl;

			// for (i = 0; i < npade+1; i++) {
			// 	ierr = MatDestroy( qpowers + i );CHKERRQ(ierr);
			// 	if (starter == "self") {
			// 		ierr = MatDestroy( qpowers_starter + i );CHKERRQ(ierr);
			// 	}
			// }
			// delete [] qpowers;
			delete_polymatrix_vector( npade+1, &qpowers );

			if (starter == "self") {
				delete_polymatrix_vector( npade+1, &qpowers_starter );
				// delete [] qpowers_starter;
			}
			if (attnfile.length() == 0) {
				atm_profile_2d->remove_property( "_ALPHA_" );
			}
			delete [] r;
			delete [] zgi_r;
			NCPA::free_cmatrix( tl, NZ, NR-1 );
		}
		
		atm_profile_2d->remove_property( "_CEFF_" );
		atm_profile_2d->remove_property( "_WC_" );
	}

	ierr = MatDestroy( &B );       CHKERRQ(ierr);
	ierr = MatDestroy( &C );       CHKERRQ(ierr);
	ierr = VecDestroy( &psi_o );   CHKERRQ(ierr);
	ierr = VecDestroy( &Bpsi_o );  CHKERRQ(ierr);
	ierr = KSPDestroy( &ksp );     CHKERRQ(ierr);
	
	delete [] k;
	delete [] n;
	delete [] c;
	delete [] a_t;
	delete [] contents;
	delete [] indices;

	return 1;
}

int NCPA::EPadeSolver::build_operator_matrix_without_topography( 
	int NZvec, double *zvec, double k0, double h2, 
	std::complex<double> impedence_factor, std::complex<double> *n, size_t nqp, 
	int boundary_index, Mat *q ) {

	// Mat q;
	PetscInt Istart, Iend, col[3];
	PetscBool FirstBlock = PETSC_FALSE, LastBlock = PETSC_FALSE;
	PetscErrorCode ierr;
	PetscScalar value[3];
	PetscInt i;

	// Set up matrices
	ierr = MatCreateSeqAIJ( PETSC_COMM_SELF, NZvec, NZvec, 3, NULL, q );CHKERRQ(ierr);
	ierr = MatSetFromOptions( *q );CHKERRQ(ierr);

	// populate
	//double bnd_cnd = -1.0 / h2;    // @todo add hook for alternate boundary conditions
	//double bnd_cnd = -2.0 / h2;      // pressure release condition
	std::complex<double> bnd_cnd = (impedence_factor * std::sqrt( h2 ) - 1.0) / h2;
	double k02 = k0*k0;
	
	ierr = MatGetOwnershipRange(*q,&Istart,&Iend);CHKERRQ(ierr);
	if (Istart==0) FirstBlock=PETSC_TRUE;
    if (Iend==NZ) LastBlock=PETSC_TRUE;
    value[0]=1.0 / h2 / k02; value[2]=1.0 / h2 / k02;
    for( i=(FirstBlock? Istart+1: Istart); i<(LastBlock? Iend-1: Iend); i++ ) {
    		if (i < boundary_index)  {
    			value[ 0 ] = 0.0;
    			value[ 1 ] = 0.0;  
    			value[ 2 ] = 0.0;
    		} else if (i == boundary_index) {
    			value[ 0 ] = 0.0;
    			value[ 1 ] = bnd_cnd/k02 + (n[i]*n[i] - 1);
    			value[ 2 ] = 1.0 / h2 / k02;
    		} else {
    			value[ 0 ] = 1.0 / h2 / k02;
    			value[ 1 ] = -2.0/h2/k02 + (n[i]*n[i] - 1);
    			value[ 2 ] = 1.0 / h2 / k02;
    		}
		    col[0]=i-1; col[1]=i; col[2]=i+1;
		    ierr = MatSetValues(*q,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (LastBlock) {
		    i=NZ-1; col[0]=NZ-2; col[1]=NZ-1;
		    value[ 0 ] = 1.0 / h2 / k02;
		    value[ 1 ] = -2.0/h2/k02 + (n[i]*n[i] - 1);
		    ierr = MatSetValues(*q,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (FirstBlock) {
		    i=0; col[0]=0; col[1]=1; 
		    if (i < boundary_index)  {
    			value[ 0 ] = 0.0;
    			value[ 1 ] = 0.0;
    		} else {
    			value[ 0 ] = bnd_cnd/k02 + (n[i]*n[i] - 1);
    			value[ 1 ] = 1.0 / h2 / k02;
    		}
		    //value[0]=bnd_cnd/k02 + (n[i]*n[i] - 1); 
		    //value[1]=1.0 / h2 / k02;
		    ierr = MatSetValues(*q,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(*q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    return 1;
}

int NCPA::EPadeSolver::solve_with_topography() {
	int i;
	std::complex<double> I( 0.0, 1.0 );
	PetscErrorCode ierr;
	PetscInt *indices;
	PetscScalar hank, *contents;
	Mat B, C, q, q_starter;
	Mat *qpowers = PETSC_NULL, *qpowers_starter = PETSC_NULL;
	Vec psi_o, Bpsi_o; //, psi_temp;
	KSP ksp;
	// PC pc;

	// set up z grid for flat ground.  When we add terrain we will need to move this inside
	// the range loop
	//int profile_index = atm_profile_2d->get_profile_index( 0.0 );
	int profile_index;
	double minlimit, maxlimit;
	atm_profile_2d->get_maximum_altitude_limits( minlimit, maxlimit );
	z_max = NCPA::min( z_max, minlimit );    // lowest valid top value
	int ground_index = 0;
	std::complex<double> ground_impedence_factor( 0.0, 0.0 );

	// truncate multiprop file if needed
	if (multiprop) {
		std::ofstream ofs( NCPAPROP_EPADE_PE_FILENAME_2D, std::ofstream::out | std::ofstream::trunc );
		ofs.close();
	}

	// truncate 1-D if necessary
	if (broadband) {
		std::ofstream ofs( NCPAPROP_EPADE_PE_FILENAME_1D, std::ofstream::out | std::ofstream::trunc );
		ofs.close();
	}

	/* @todo move this into constructor as much as possible */
	// if (use_topo) {
		z_bottom = -5000.0;    // make this eventually depend on frequency
		z_bottom -= fmod( z_bottom, dz );
		z_ground = atm_profile_2d->get( 0.0, "Z0" );
		NZ = ((int)std::floor((z_max - z_bottom) / dz)) + 1;
		z = new double[ NZ ];
		z_abs = new double[ NZ ];
		std::memset( z, 0, NZ * sizeof( double ) );
		std::memset( z_abs, 0, NZ * sizeof( double ) );
		indices = new PetscInt[ NZ ];
		std::memset( indices, 0, NZ*sizeof(PetscInt) );
		for (i = 0; i < NZ; i++) {
			z[ i ] = ((double)i) * dz + z_bottom;
			z_abs[ i ] = z[ i ];
			indices[ i ] = i;
		}
		zs = NCPA::max( zs, z_ground );

		// define ground_index, which is J in @notes
		ground_index = NCPA::find_closest_index( z, NZ, z_ground );
		if ( z[ ground_index ] < z_ground ) {
			ground_index++;
		}

	// } else {
	// 	atm_profile_2d->get_minimum_altitude_limits( minlimit, z_min );
	// 	//z_min = atm_profile_2d->get_highest_minimum_altitude();
	// 	if ( (!z_ground_specified) && atm_profile_2d->contains_scalar( 0.0, "Z0" )) {
	// 		z_ground = atm_profile_2d->get( 0.0, "Z0" );
	// 	}
	// 	if (z_ground < z_min) {
	// 		std::cerr << "Supplied ground height is outside of atmospheric specification." << std::endl;
	// 		exit(0);
	// 	}
	//   	z_bottom = z_min;
	// 	// fill and convert to SI units
	// 	//double dz       = (z_max - z_ground)/(NZ - 1);	// the z-grid spacing
	// 	NZ = ((int)std::floor((z_max - z_ground) / dz)) + 1;
	// 	z = new double[ NZ ];
	// 	z_abs = new double[ NZ ];
	// 	std::memset( z, 0, NZ * sizeof( double ) );
	// 	std::memset( z_abs, 0, NZ * sizeof( double ) );
	// 	indices = new PetscInt[ NZ ];
	// 	std::memset( indices, 0, NZ*sizeof(PetscInt) );
	// 	for (i = 0; i < NZ; i++) {
	// 		z[ i ]     = ((double)i) * dz;
	// 		z_abs[ i ] = z[ i ] + z_ground;
	// 		indices[ i ] = i;
	// 	}
	// 	zs = NCPA::max( zs-z_ground+dz, dz );

	// 	// ground_index is already 0
	// }
	
	// constants for now
	double h = dz;
	double h2 = h * h;
	double dr;

	// set up for source atmosphere
	double k0 = 0.0, c0 = 0.0;
	double *c = new double[ NZ ];
	double *a_t = new double[ NZ ];
	std::complex<double> *k = new std::complex<double>[ NZ ];
	std::complex<double> *n = new std::complex<double>[ NZ ];

	// write broadband header for testing
	if (broadband) {
		write_broadband_header( NCPAPROP_EPADE_PE_FILENAME_BROADBAND, azi, NAz, f, NF, 1.0e8 );
	}

	// freq and calc_az hold the current values of azimuth and frequency, respectively
	// these are used in the output routines, so make sure they get set correctly
	// whenever you change frequencies and azimuths
	for (int azind = 0; azind < NAz; azind++) {
		calc_az = azi[ azind ];
		std::cout << "Infrasound PE code at f = " << freq << " Hz, azi = " 
			<< calc_az << " deg" << std::endl;

		profile_index = -1;
		atm_profile_2d->calculate_wind_component( "_WC_", "_WS_", "_WD_", calc_az );
		atm_profile_2d->calculate_effective_sound_speed( "_CEFF_", "_C0_", "_WC_" );

		for (int freqind = 0; freqind < NF; freqind++) {

			freq = f[ freqind ];

			// calculate attenuation as a function of frequency if not externally supplied
			if (attnfile.length() == 0) {
				atm_profile_2d->calculate_attenuation( "_ALPHA_", "T", "P", "RHO", freq );
			}

			// Set up range vector
			if (NR_requested == 0) {
		  		dr = 340.0 / freq;
				NR = (int)ceil( r_max / dr );
		  	} else {
		  		NR = NR_requested;
		  		dr = r_max / NR;
		  	}
		  	r = new double[ NR ];
		  	std::memset( r, 0, NR * sizeof(double) );
		  	zgi_r = new int[ NR ];
		  	std::memset( zgi_r, 0, NR * sizeof( int ) );
		  	int i;
		  	for (i = 0; i < NR; i++) {
		  		r[ i ] = ((double)(i+1)) * dr;
		  	}

		  	// set up transmission loss matrix
			tl = NCPA::cmatrix( NZ, NR-1 );

			// calculate ground impedence (script A in notes in eq. 12)
			double rho0 = atm_profile_2d->get( 0.0, "RHO", z_ground );
			double lambBC = atm_profile_2d->get_first_derivative( 0.0, "RHO", z_ground ) / (2.0 * rho0);
			//lambBC = 0.0;
			if (user_ground_impedence_found) {
				ground_impedence_factor = I * 2.0 * PI * freq * rho0 / user_ground_impedence + lambBC;
				std::cout << "Using user ground impedence of " << user_ground_impedence << std::endl;
			} else {
				ground_impedence_factor.real( lambBC );
				ground_impedence_factor.imag( 0.0 );
				std::cout << "Using default rigid ground with Lamb BC" << std::endl;
			}

			calculate_atmosphere_parameters( atm_profile_2d, NZ, z, 0.0, z_ground, lossless, 
				top_layer, freq, use_topo, k0, c0, c, a_t, k, n );
			std::cout << "k0 = " << k0 << std::endl;

			// output c and rho to files for testing
			std::ofstream cfile("effective_c.dat");
			std::ofstream rfile("rho.dat");
			for (int iii = 0; iii < NZ; iii++) {
				cfile << c[ iii ] << std::endl;
				if (z[iii] >= 0.0) {
					rfile << z[iii] << " " << atm_profile_2d->get( 0.0, "RHO", z[iii] ) << std::endl;
				}
			}
			cfile.close();
			rfile.close();

			// calculate q matrices.  we need to redo this every time we change the atmosphere, but
			// can keep reusing the precalculated powers until that point
			// qpowers = new Mat[ npade+1 ];
			// make_q_powers( atm_profile_2d, NZ, z, 0.0, k, k0, h2, z_ground, ground_impedence_factor, 
			// 	n, npade+1, ground_index, 0.0, PETSC_NULL, qpowers );
			build_operator_matrix_with_topography( atm_profile_2d, NZ, z, 0.0, k, k0, h2, 
				z_ground, ground_impedence_factor, n, ground_index, PETSC_NULL, &q );
			create_polymatrix_vector( npade+1, &q, &qpowers );
			ierr = MatDestroy( &q );CHKERRQ(ierr);


			if (starter == "self") {
				// qpowers_starter = new Mat[ npade+1 ];
				// make_q_powers( atm_profile_2d, NZ, z, 0.0, k, k0, h2, z_ground, 
				// 	ground_impedence_factor, n, npade+1, ground_index, 0.0, PETSC_NULL, qpowers_starter );
				build_operator_matrix_with_topography( atm_profile_2d, NZ, z, 0.0, k, k0, 
					h2, z_ground, ground_impedence_factor, n, ground_index, PETSC_NULL, 
					&q_starter );
				create_polymatrix_vector( npade+1, &q_starter, &qpowers_starter );
				
				// Output Q[0] for testing of the starter
				// PetscInt ncols;
				// const PetscScalar *colvals;
				// for (PetscInt rownum = 0; rownum < NZ; rownum++) {
				// 	ierr = MatGetRow( qpowers_starter[ 0 ], rownum, &ncols, NULL, &colvals );CHKERRQ(ierr);
				// 	std::cout << z[ rownum ] << " " << ncols << " ";
				// 	for (PetscInt colnum = 0; colnum < ncols; colnum++) {
				// 		std::cout << colvals[colnum].real() << " " << colvals[colnum].imag()
				// 				  << " ";
				// 	}
				// 	std::cout << std::endl;
				// 	ierr = MatRestoreRow( qpowers_starter[ 0 ], rownum, &ncols, NULL, &colvals );CHKERRQ(ierr);
				// }
				//get_starter_self( NZ, z, zs, z_ground, k0, qpowers_starter, npade, &psi_o );
				get_starter_self_revised( NZ, z, zs, r[ 0 ], z_ground, k0, qpowers_starter, 
					npade, &psi_o );
				ierr = MatDestroy( &q_starter );CHKERRQ(ierr);
			} else if (starter == "gaussian") {
				qpowers_starter = qpowers;
				get_starter_gaussian( NZ, z, zs, k0, ground_index, &psi_o );
			} else {
				std::cerr << "Unrecognized starter type: " << starter << std::endl;
				exit(0);
			}

			std::cout << "Outputting starter..." << std::endl;
			outputVec( psi_o, z, NZ, "starter_topo.dat" );

			std::cout << "Finding ePade coefficients..." << std::endl;
			std::vector< PetscScalar > P, Q;
			std::vector< PetscScalar > taylor = taylor_exp_id_sqrt_1pQ_m1( 2*npade, k0*dr );
			calculate_pade_coefficients( &taylor, npade, npade+1, &P, &Q );
			generate_polymatrices( qpowers_starter, npade, NZ, P, Q, &B, &C );

			std::cout << "Marching out field..." << std::endl;
			ierr = VecDuplicate( psi_o, &Bpsi_o );CHKERRQ(ierr);
			contents = new PetscScalar[ NZ ];

			ierr = KSPCreate( PETSC_COMM_SELF, &ksp );CHKERRQ(ierr);
			ierr = KSPSetOperators( ksp, C, C );CHKERRQ(ierr);
			ierr = KSPSetFromOptions( ksp );CHKERRQ(ierr);
			for (PetscInt ir = 0; ir < (NR-1); ir++) {

				double rr = r[ ir ];
				z_ground = atm_profile_2d->get_interpolated_ground_elevation( rr );
				// check for atmosphere change
				//if (((int)(atm_profile_2d->get_profile_index( rr ))) != profile_index) {
				
					//profile_index = atm_profile_2d->get_profile_index( rr );
					calculate_atmosphere_parameters( atm_profile_2d, NZ, z, rr, z_ground, lossless, 
						top_layer, freq, use_topo, k0, c0, c, a_t, k, n );
					Mat last_q;
					ierr = MatConvert( qpowers[0], MATSAME, MAT_INITIAL_MATRIX, &last_q );CHKERRQ(ierr);
					delete_polymatrix_vector( npade+1, &qpowers );
					// for (i = 0; i < npade+1; i++) {
					// 	ierr = MatDestroy( qpowers + i );CHKERRQ(ierr);
					// }
					// @todo recalculate z_ground for the new r when we do undulating ground
					build_operator_matrix_with_topography( atm_profile_2d, NZ, z, rr, k, 
						k0, h2, z_ground, ground_impedence_factor, n, ground_index, 
						last_q, &q );
					create_polymatrix_vector( npade+1, &q, &qpowers );
					// make_q_powers( atm_profile_2d, NZ, z, rr, k, k0, h2, z_ground, 
					// 	ground_impedence_factor, n, npade+1, ground_index, 
					// 	atm_profile_2d->get_interpolated_ground_elevation_first_derivative( rr ), 
					// 	last_q, qpowers );
					ierr = MatDestroy( &last_q );CHKERRQ(ierr);
					ierr = MatDestroy( &q );CHKERRQ(ierr);

					// epade( npade, k0, dr, &P, &Q );
					// std::cout << "Original P:" << std::endl;
					// printSTLVector( &P );
					// std::cout << "Original Q:" << std::endl;
					// printSTLVector( &Q );
					taylor = taylor_exp_id_sqrt_1pQ_m1( 2*npade, k0*dr );
					calculate_pade_coefficients( &taylor, npade, npade+1, &P, &Q );
					// std::cout << "New P:" << std::endl;
					// printSTLVector( &P );
					// std::cout << "New Q:" << std::endl;
					// printSTLVector( &Q );

					ierr = MatZeroEntries( B );CHKERRQ(ierr);
					ierr = MatZeroEntries( C );CHKERRQ(ierr);
					generate_polymatrices( qpowers, npade, NZ, P, Q, &B, &C );
					// std::cout << "Switching to atmosphere index " << profile_index 
					// 	<< " at range = " << rr/1000.0 << " km" << std::endl;
				//}



				hank = sqrt( 2.0 / ( PI * k0 * rr ) ) * exp( I * ( k0 * rr - PI/4.0 ) );
				ierr = VecGetValues( psi_o, NZ, indices, contents );
				for (i = 0; i < NZ; i++) {
					tl[ i ][ ir ] = contents[ i ] * hank;
				}

				// if (use_topo) {
					//double z0g = atm_profile_2d->get( rr, "Z0" );
					// double z0g = atm_profile_2d->get_interpolated_ground_elevation( rr );
					double z0g = z_ground;
					z0g = NCPA::max( z0g, zr );
					zgi_r[ ir ] = NCPA::find_closest_index( z, NZ, z0g );
					if ( z[ zgi_r[ ir ] ] < z0g ) {
						zgi_r[ ir ]++;
					}
					//zgi_r[ ir ]++;
				// } else {
				// 	zgi_r[ ir ] = 0;
				// }


				if ( fmod( rr, 1.0e5 ) < dr) {
					std::cout << " -> Range " << rr/1000.0 << " km" << std::endl;
				}

				// Set up the system
				// C * psi_o_next = B * psi_o
				ierr = MatMult( B, psi_o, Bpsi_o );CHKERRQ(ierr);
				ierr = KSPSetOperators( ksp, C, C );CHKERRQ(ierr);
				ierr = KSPSolve( ksp, Bpsi_o, psi_o );CHKERRQ(ierr);
			}
			std::cout << "Stopped at range " << r[ NR-1 ]/1000.0 << " km" << std::endl;

			

			if (multiprop) {
				if (write1d) {
					std::cout << "Writing 1-D output to " << NCPAPROP_EPADE_PE_FILENAME_MULTIPROP << std::endl;
					output1DTL( NCPAPROP_EPADE_PE_FILENAME_MULTIPROP, true );
				}
			} else { 
				if (write1d) {
					std::cout << "Writing 1-D output to " << NCPAPROP_EPADE_PE_FILENAME_1D << std::endl;
					output1DTL( NCPAPROP_EPADE_PE_FILENAME_1D, broadband );
				}
				if (write2d) {
					std::cout << "Writing 2-D output to " << NCPAPROP_EPADE_PE_FILENAME_2D << std::endl;
					output2DTL( NCPAPROP_EPADE_PE_FILENAME_2D );
				}
			}

			// write broadband body for testing
			if (broadband) {
				write_broadband_results( NCPAPROP_EPADE_PE_FILENAME_BROADBAND, calc_az, freq, r, NR, z_abs, NZ, tl, 1.0e8 );
			}
			
			std::cout << std::endl;

			delete_polymatrix_vector( npade+1, &qpowers );
			// for (i = 0; i < npade+1; i++) {
			// 	ierr = MatDestroy( qpowers + i );CHKERRQ(ierr);
			// 	if (starter == "self") {
			// 		ierr = MatDestroy( qpowers_starter + i );CHKERRQ(ierr);
			// 	}
			// }
			// delete [] qpowers;
			if (starter == "self") {
				delete_polymatrix_vector( npade+1, &qpowers_starter );
				// delete [] qpowers_starter;
			}
			if (attnfile.length() == 0) {
				atm_profile_2d->remove_property( "_ALPHA_" );
			}
			delete [] r;
			delete [] zgi_r;
			NCPA::free_cmatrix( tl, NZ, NR-1 );
		}
		
		atm_profile_2d->remove_property( "_CEFF_" );
		atm_profile_2d->remove_property( "_WC_" );
	}

	ierr = MatDestroy( &B );       CHKERRQ(ierr);
	ierr = MatDestroy( &C );       CHKERRQ(ierr);
	ierr = VecDestroy( &psi_o );   CHKERRQ(ierr);
	ierr = VecDestroy( &Bpsi_o );  CHKERRQ(ierr);
	ierr = KSPDestroy( &ksp );     CHKERRQ(ierr);
	
	delete [] k;
	delete [] n;
	delete [] c;
	delete [] a_t;
	delete [] contents;
	delete [] indices;

	return 1;
}

// create a vector of powers of a given matrix
int NCPA::EPadeSolver::create_polymatrix_vector( size_t nterms, const Mat *Q, Mat **qpowers ) {

	PetscErrorCode ierr;
	PetscInt i;

	if ((*qpowers) != PETSC_NULL) {
		delete_polymatrix_vector( nterms, qpowers );
	}

	*qpowers = new Mat[ nterms ];
	ierr = MatConvert( *Q, MATSAME, MAT_INITIAL_MATRIX, *qpowers );CHKERRQ(ierr);
	for (i = 1; i < (PetscInt)nterms; i++) {
		ierr = MatCreate( PETSC_COMM_SELF, (*qpowers) + i );CHKERRQ(ierr);
		ierr = MatSetFromOptions( (*qpowers)[ i ] );CHKERRQ(ierr);
		ierr = MatMatMult( (*qpowers)[i-1], (*qpowers)[0], MAT_INITIAL_MATRIX, PETSC_DEFAULT, 
			(*qpowers) + i );CHKERRQ(ierr);
	}

	return 1;
}

// clean up a vector of powers of a matrix
int NCPA::EPadeSolver::delete_polymatrix_vector( size_t nterms, Mat **qpowers ) {
	PetscErrorCode ierr;
	if ((*qpowers) != PETSC_NULL) {
		for (size_t i = 0; i < nterms; i++) {
			ierr = MatDestroy( (*qpowers) + i ); CHKERRQ(ierr);
		}
		delete [] *qpowers;
		*qpowers = PETSC_NULL;
	}

	return 1;
}

// Calculate and return k0, c0, c, a, k, and n
void NCPA::EPadeSolver::calculate_atmosphere_parameters( NCPA::Atmosphere2D *atm, int NZvec, double *z_vec, 
	double r, double z_g, bool use_lossless, bool use_top_layer, double freq, bool absolute, 
	double &k0, double &c0, double *c_vec, double *a_vec, std::complex<double> *k_vec, 
	std::complex<double> *n_vec ) {

	std::complex<double> I( 0.0, 1.0 );

	std::memset( c_vec, 0, NZvec * sizeof(double) );
	std::memset( a_vec, 0, NZvec * sizeof(double) );
	std::memset( k_vec, 0, NZvec * sizeof( std::complex< double > ) );
	std::memset( n_vec, 0, NZvec * sizeof( std::complex< double > ) );

	// z_vec is relative to ground
	if (absolute) {
		fill_atm_vector_absolute( atm, r, NZvec, z_vec, "_CEFF_", c_underground, c_vec );
	} else {
		fill_atm_vector_relative( atm, r, NZvec, z_vec, "_CEFF_", z_g, c_vec );
	}
	c0 = atm->get( r, "_CEFF_", z_g );

	if (!use_lossless) {
		if (absolute) {
			fill_atm_vector_absolute( atm, r, NZvec, z_vec, "_ALPHA_", 0.0, a_vec );
		} else {
			fill_atm_vector_relative( atm, r, NZvec, z_vec, "_ALPHA_", z_g, a_vec );
		}
	}
	double *abslayer = new double[ NZvec ];
	memset( abslayer, 0, NZvec * sizeof(double) );
	if (use_top_layer) {
		absorption_layer( c0 / freq, z_vec, NZvec, abslayer );
	}
	
	k0 = 2.0 * PI * freq / c0;
	
	// Set up vectors
	for (int i = 0; i < NZvec; i++) {
		if (absolute && (z_vec[i] < z_g)) {
			k_vec[ i ] = 0.0;    // k == 0 below the ground
		} else {
			k_vec[ i ] = 2.0 * PI * freq / c_vec[ i ] + (a_vec[ i ] + abslayer[ i ]) * I;
			//k_vec[ i ] = 2.0 * PI * freq / c_vec[ i ] + a_vec[ i ] * I;
		}
		n_vec[ i ] = k_vec[ i ] / k0;
	}
}

void NCPA::EPadeSolver::fill_atm_vector_relative( NCPA::Atmosphere2D *atm, double range, int NZvec, double *zvec, 
	std::string key, double groundheight, double *vec ) {

	for (int i = 0; i < NZvec; i++) {
		vec[i] = atm->get( range, key, zvec[i] + groundheight );
	}
}

void NCPA::EPadeSolver::fill_atm_vector_absolute( NCPA::Atmosphere2D *atm, double range, int NZvec, double *zvec, 
	std::string key, double fill_value, double *vec ) {

	//double zmin = atm->get( range, "Z0" );
	double zmin = atm->get_interpolated_ground_elevation( range ); 

	// double bound_val = atm->get( range, key, zmin );
	// double tempval;
	for (int i = 0; i < NZvec; i++) {
		// if (zvec[i] < (zmin - 500.0)) {
		if (zvec[i] < zmin) {
			vec[i] = fill_value;
		// } else if (zvec[i] < zmin) {
		// 	double factor = 0.5 - 0.5 * std::cos( PI * (zmin - zvec[i]) / 500.0  );
		// 	tempval = (fill_value - bound_val) * factor + bound_val;
		// 	vec[i] = (fill_value - bound_val) * factor + bound_val;
		} else {
			vec[i] = atm->get( range, key, zvec[i] );
		}
	}
}


int NCPA::EPadeSolver::generate_polymatrix( Mat *qpowers, int qpowers_size, int NZ, 
			std::vector< std::complex< double > > &T, Mat *B ) {

	PetscErrorCode ierr;
	PetscInt Istart, Iend, i;
	PetscScalar value;

	ierr = MatCreateSeqAIJ( PETSC_COMM_SELF, NZ, NZ, 2*T.size()+1, NULL, B );CHKERRQ(ierr);
	ierr = MatSetFromOptions( *B );CHKERRQ(ierr);

	// start B off as T[0]
	ierr = MatGetOwnershipRange(*B,&Istart,&Iend);CHKERRQ(ierr);
	value = T[0];
	for (i = Istart; i < Iend; i++) {
		ierr = MatSetValues( *B, 1, &i, 1, &i, &value, INSERT_VALUES );CHKERRQ(ierr);
	}
	ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscInt nterms = NCPA::min( T.size(), qpowers_size );
	for (i = 1; i < (PetscInt)nterms; i++) {
		ierr = MatAXPY( *B, T[ i ], qpowers[ i-1 ], DIFFERENT_NONZERO_PATTERN );CHKERRQ(ierr);
	}
	return 1;
}


int NCPA::EPadeSolver::generate_polymatrices( Mat *qpowers, int npade, int NZ, 
	std::vector< std::complex< double > > &P, std::vector< std::complex< double > > &Q,
	Mat *B, Mat *C ) {

	PetscErrorCode ierr;
	PetscInt Istart, Iend, i;
	PetscScalar value;

	ierr = MatCreateSeqAIJ( PETSC_COMM_SELF, NZ, NZ, 2*npade-1, NULL, B );CHKERRQ(ierr);
	ierr = MatSetFromOptions( *B );CHKERRQ(ierr);
	ierr = MatCreateSeqAIJ( PETSC_COMM_SELF, NZ, NZ, 2*npade+1, NULL, C );CHKERRQ(ierr);
	ierr = MatSetFromOptions( *C );CHKERRQ(ierr);

	ierr = MatGetOwnershipRange(*B,&Istart,&Iend);CHKERRQ(ierr);
	// by definition Q[0] is 1.  It so happens that P[0] is also 1, but this is not 
	// guaranteed.
	// @todo generalize this fot the case where P[0] != 1
	value = 1.0;
	for (i = Istart; i < Iend; i++) {
		ierr = MatSetValues( *B, 1, &i, 1, &i, &value, INSERT_VALUES );CHKERRQ(ierr);
	}
	ierr = MatGetOwnershipRange( *C, &Istart, &Iend );CHKERRQ(ierr);
	for (i = Istart; i < Iend; i++) {
		ierr = MatSetValues( *C, 1, &i, 1, &i, &value, INSERT_VALUES );CHKERRQ(ierr);
	}

	ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(*C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

	for (i = 1; i < (PetscInt)(Q.size()); i++) {
		ierr = MatAXPY( *C, Q[ i ], qpowers[ i-1 ], DIFFERENT_NONZERO_PATTERN );CHKERRQ(ierr);
	}
	for (i = 1; i < (PetscInt)(P.size()); i++) {
		ierr = MatAXPY( *B, P[ i ], qpowers[ i-1 ], DIFFERENT_NONZERO_PATTERN );CHKERRQ(ierr);
	}
	return 1;
}

int NCPA::EPadeSolver::build_operator_matrix_with_topography( NCPA::Atmosphere2D *atm, 
	int NZvec, double *zvec, double r, std::complex<double> *k, double k0, double h2, 
	double z_s, std::complex<double> impedence_factor, std::complex<double> *n, 
	int boundary_index, const Mat &last_q, Mat *q ) {

	//Mat q;
	PetscInt Istart, Iend, *col, *indices;
	PetscInt *nonzeros;
	PetscBool FirstBlock = PETSC_FALSE, LastBlock = PETSC_FALSE;
	PetscErrorCode ierr;
	PetscScalar value[3], *rowDiff;
	PetscInt i, j;

	// Calculate parameters
	double h = std::sqrt( h2 );
	double J_s = (z_s - zvec[0]) / h;
	int Ji = NCPA::find_closest_index( zvec, NZvec, z_s );  // first index above ground
	if ( zvec[ Ji ] < z_s ) {
		Ji++;
	}
	double dJ = (double)Ji;
	//double dJ = zvec[ Ji ] / h;

	// number of nonzero values
	nonzeros = new PetscInt[ NZvec ];
	indices = new PetscInt[ NZvec ];
	col = new PetscInt[ NZvec ];
	for (i = 0; i < NZvec; i++) {
		nonzeros[ i ] = 3;
		indices[ i ] = i;
		col[ i ] = 0;
	}
	nonzeros[ 0 ] = 2;
	nonzeros[ NZvec-1 ] = 2;


	// calculate intermediate variables as shown in notes
	double rho_a = atm->get( r, "RHO", z_s );
	double Gamma = 0.5 * atm->get_first_derivative( r, "RHO", z_s ) / rho_a;
	double rho_b = rho_a * 1000.0;
	double Anom = (1.0 / rho_a) * (1.0 / (dJ - J_s));
	double Bnom = (1.0 / rho_b) * (1.0 / (J_s - dJ + 1.0));
	double denom = Anom + Bnom - (h * Gamma);
	double s_A = Anom / denom;
	double s_B = Bnom / denom;
	double a, b, c, alpha, beta, gamma;
	a   = s_B / (dJ - J_s);
	b   = (s_A - 2.0) / (dJ - J_s);
	c   = 1.0 / (dJ - J_s);
	alpha = 1.0 / (J_s - dJ + 1.0);
	beta  = (s_B - 2.0) / (J_s - dJ + 1.0);
	gamma = s_A / (J_s - dJ + 1.0);
	
	// Calculate matrix ratio representation of sqrt(1+Q)
	rowDiff = new PetscScalar[ NZvec ];
	if (last_q != PETSC_NULL) {
		PetscScalar I( 0.0, 1.0 ), *rowAbove, *rowBelow;
		PetscScalar M = I * k0 * atm->get_interpolated_ground_elevation_first_derivative( r ) * h / denom;
		Vec vecAbove, vecBelow;
		PetscInt num_nonzeros;
		approximate_sqrt_1pQ( NZvec, &last_q, Ji, &vecBelow, &vecAbove, &num_nonzeros );
		ierr = VecScale( vecBelow, M );CHKERRQ(ierr);
		ierr = VecScale( vecAbove, M );CHKERRQ(ierr);
		
		// get the Ji'th and (Ji-1)'th rows of the M matrix
		rowBelow = new PetscScalar[ NZvec ];
		std::memset( rowBelow, 0, NZvec * sizeof(PetscScalar) );
		ierr = VecGetValues( vecBelow, NZvec, indices, rowBelow );CHKERRQ(ierr);
		// nonzeros[ Ji-1 ] = NZvec;
		nonzeros[ Ji-1 ] = num_nonzeros;

		rowAbove = new PetscScalar[ NZvec ];
		std::memset( rowAbove, 0, NZvec * sizeof(PetscScalar) );
		ierr = VecGetValues( vecAbove, NZvec, indices, rowAbove );CHKERRQ(ierr);
		// nonzeros[ Ji ] = NZvec;
		nonzeros[ Ji ] = num_nonzeros;
		//std::cout << num_nonzeros << " nonzeros expected" << std::endl;

		for (i = 0; i < NZvec; i++) {
			rowDiff[ i ] = rowAbove[ i ] - rowBelow[ i ];
		}

		delete [] rowAbove;
		delete [] rowBelow;
		ierr = VecDestroy( &vecAbove );CHKERRQ(ierr);
		ierr = VecDestroy( &vecBelow );CHKERRQ(ierr);
	} 

	// Set up matrices
	ierr = MatCreateSeqAIJ( PETSC_COMM_SELF, NZvec, NZvec, 0, nonzeros, q );CHKERRQ(ierr);
	// ierr = MatCreateSeqAIJ( PETSC_COMM_SELF, NZvec, NZvec, 3, PETSC_NULL, q );CHKERRQ(ierr);
	ierr = MatSetFromOptions( *q );CHKERRQ(ierr);
	

	// populate
	//double bnd_cnd = -1.0 / h2;    // @todo add hook for alternate boundary conditions
	//double bnd_cnd = -2.0 / h2;      // pressure release condition
	//std::complex<double> bnd_cnd = (impedence_factor * std::sqrt( h2 ) - 1.0) / h2;
	double k02 = k0*k0;
	
	// If this process is being split over processors, we need to check to see
	// if this particular instance contains the first or last rows, because those
	// get filled differently
	ierr = MatGetOwnershipRange(*q,&Istart,&Iend);CHKERRQ(ierr);

	// Does this instance contain the first row?
	if (Istart == 0) {
		FirstBlock=PETSC_TRUE;
	}

	// Does this instance contain the last row?
    if (Iend==NZ) {
    	LastBlock=PETSC_TRUE;
    }

    //value[0]=1.0 / h2 / k02; value[2]=1.0 / h2 / k02;
    // iterate over block.  If this instance contains the first row, leave that one
    // for later, same for if this instance contains the last row.
    PetscScalar *Drow = new PetscScalar[ NZvec ];
    //double Drow[ 3 ];
    memset( Drow, 0, NZvec*sizeof(PetscScalar) );
    for( i=(FirstBlock? Istart+1: Istart); i<(LastBlock? Iend-1: Iend); i++ ) {

		// set column numbers.  Since the matrix Q is tridiagonal (because input 
		// matrix D is tridiagonal and K is diagonal), column indices are
		// i-1, i, i+1
    	col[ 0 ] = i-1;
    	col[ 1 ] = i;
    	col[ 2 ] = i+1;

    	// Set values.  This will be the same unless we're at the indices immediately
    	// below or above the ground surface
    	if (i == (Ji-1)) {

    		if (last_q != PETSC_NULL) {
	    		// this is the alpha, beta, gamma row
	    		std::memcpy( col, indices, NZvec*sizeof(PetscInt) );
	    		for (j = 0; j < NZvec; j++) {
	    			Drow[ j ] = -rowDiff[ j ] / ( h2 * (J_s - dJ + 1.0));
	    		}
	    		//std::memcpy( Drow, rowBelow, NZvec*sizeof(PetscScalar) );

	    		Drow[ Ji-2 ] += alpha;
	    		Drow[ Ji-1 ] += beta;
	    		Drow[  Ji  ] += gamma;
	    	} else {   // M == 0
	    		Drow[0] = alpha;
	    		Drow[1] = beta;
	    		Drow[2] = gamma;
	    	}
    		
    	} else if (i == Ji) {
    		// this is the a, b, c row
    		if (last_q != PETSC_NULL) {
	    		std::memcpy( col, indices, NZvec*sizeof(PetscInt) );
	    		for (j = 0; j < NZvec; j++) {
	    			Drow[ j ] = -rowDiff[ j ] / ( h2 * (dJ - J_s));
	    		}
	    		//std::memcpy( Drow, rowBelow, NZvec*sizeof(PetscScalar) );

	    		Drow[ Ji-1 ] += a;
	    		Drow[  Ji  ] += b;
	    		Drow[ Ji+1 ] += c;
	    	} else {
	    		Drow[0] = a;
	    		Drow[1] = b;
	    		Drow[2] = c;
	    	}
    	} else {
    		Drow[0] = 1.0;
    		Drow[1] = -2.0;
    		Drow[2] = 1.0;
    	}

    	for (j = 0; j < nonzeros[ i ]; j++) {
    		if (col[j] == i) {
    			Drow[ j ] = ( (Drow[ j ] / h2) + k[ i ]*k[ i ] - k02 ) / k02;
    		} else {
    			Drow[ j ] /= (h2 * k02);
    		}
    	}
    	// value[ 0 ] = Drow[0] / h2 / k02;
    	// value[ 1 ] = ( (Drow[1] / h2) + k[ i ]*k[ i ] - k02 ) / k02;
    	// value[ 2 ] = Drow[2] / h2 / k02;

		ierr = MatSetValues(*q,1,&i,nonzeros[ i ],col,Drow,INSERT_VALUES);CHKERRQ(ierr);
		// ierr = MatSetValues(*q,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (LastBlock) {
		    i=NZ-1; col[0]=NZ-2; col[1]=NZ-1;
		    value[ 0 ] = 1.0 / h2 / k02;
		    //value[ 1 ] = -2.0/h2/k02 + (n[i]*n[i] - 1);
		    value[ 1 ] = ( (-2.0 / h2) + k[ i ]*k[ i ] - k02 ) / k02;
		    ierr = MatSetValues(*q,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (FirstBlock) {
		    i=0; col[0]=0; col[1]=1; 
		    if (i == (Ji-1))  {
		    	if (last_q != PETSC_NULL) {
			    	std::memcpy( col, indices, NZvec*sizeof(PetscInt) );
			    	for (j = 0; j < NZvec; j++) {
		    			Drow[ j ] = -rowDiff[ j ] / ( h2 * (J_s - dJ + 1.0));
		    		}
		    		// std::memcpy( Drow, rowBelow, NZvec*sizeof(PetscScalar) );

		    		Drow[ 0 ] += beta;
		    		Drow[ 1 ] += gamma;
		    	} else {
		    		Drow[ 0 ] = beta;
		    		Drow[ 1 ] = gamma;
		    	}
		    	//Drow[0] = alpha;
	    		// Drow[1] = beta;
	    		// Drow[2] = gamma;
    			// value[ 0 ] = 0.0;
    			// value[ 1 ] = 0.0;
    		} else if (i == Ji) {
    			if (last_q != PETSC_NULL) {
	    			std::memcpy( col, indices, NZvec*sizeof(PetscInt) );
	    			for (j = 0; j < NZvec; j++) {
		    			Drow[ j ] = -rowDiff[ j ] / ( h2 * (dJ - J_s));
		    		}
		    		// std::memcpy( Drow, rowBelow, NZvec*sizeof(PetscScalar) );

		    		Drow[ 0 ] += b;
		    		Drow[ 1 ] += c;
		    	} else {
		    		Drow[ 0 ] = b;
		    		Drow[ 1 ] = c;
		    	}
	    			// Drow[0] = a;
		    		// Drow[1] = b;
		    		// Drow[2] = c;
    		} else {
    			//Drow[0] = 1.0;
	    		Drow[0] = -2.0;
	    		Drow[1] = 1.0;
    		}
    		for (j = 0; j < nonzeros[ i ]; j++) {
	    		if (col[j] == i) {
	    			Drow[ j ] = ( (Drow[ j ] / h2) + k[ i ]*k[ i ] - k02 ) / k02;
	    		} else {
	    			Drow[ j ] /= (h2 * k02);
	    		}
	    	}

    		// value[ 0 ] = ( (Drow[1] / h2) + k[ i ]*k[ i ] - k02 ) / k02;
    		// value[ 1 ] = Drow[2] / h2 / k02;
		    ierr = MatSetValues(*q,1,&i,nonzeros[i],col,Drow,INSERT_VALUES);CHKERRQ(ierr);
		    // ierr = MatSetValues(*q,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(*q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    delete [] nonzeros;
    delete [] indices;
    delete [] col;
    delete [] Drow;
    delete [] rowDiff;
	return 1;
}

int NCPA::EPadeSolver::approximate_sqrt_1pQ( int NZvec, const Mat *Q, PetscInt Ji, Vec *vecBelow, Vec *vecAbove, PetscInt *nonzeros ) {

	PetscErrorCode ierr;
	

	/* for order (0,0) */

	// const PetscScalar diag = 1;
	// const PetscInt diagIndBelow = Ji-1;
	// const PetscInt diagIndAbove = Ji;
	// ierr = VecCreate( PETSC_COMM_SELF, vecBelow );CHKERRQ(ierr);
	// ierr = VecSetType( *vecBelow, VECSEQ );CHKERRQ(ierr);
	// ierr = VecSetFromOptions( *vecBelow );CHKERRQ(ierr);
	// ierr = VecSetSizes( *vecBelow, PETSC_DECIDE, NZvec );CHKERRQ(ierr);
	// ierr = VecSet( *vecBelow, 0 );CHKERRQ(ierr);
	// ierr = VecSetValues( *vecBelow, 1, &diagIndBelow, &diag, INSERT_VALUES );CHKERRQ(ierr);
	// ierr = VecCreate( PETSC_COMM_SELF, vecAbove );CHKERRQ(ierr);
	// ierr = VecSetType( *vecAbove, VECSEQ );CHKERRQ(ierr);
	// ierr = VecSetFromOptions( *vecAbove );CHKERRQ(ierr);
	// ierr = VecSetSizes( *vecAbove, PETSC_DECIDE, NZvec );CHKERRQ(ierr);
	// ierr = VecSet( *vecAbove, 0 );CHKERRQ(ierr);
	// ierr = VecSetValues( *vecBelow, 1, &diagIndAbove, &diag, INSERT_VALUES );CHKERRQ(ierr);
	// ierr = VecAssemblyBegin( *vecBelow );CHKERRQ(ierr);
	// ierr = VecAssemblyEnd( *vecBelow );CHKERRQ(ierr);
	// ierr = VecAssemblyBegin( *vecAbove );CHKERRQ(ierr);
	// ierr = VecAssemblyEnd( *vecBelow );CHKERRQ(ierr);
	// *nonzeros = 3;
	// return 1;

	/* For order (1,0) */
	
	PetscInt nvals;
	const PetscInt *indices;
	Mat halfQ;
	const PetscScalar *values;
	const PetscScalar diag = 1;
	const PetscInt diagIndBelow = Ji-1;
	const PetscInt diagIndAbove = Ji;
	
	ierr = VecCreate( PETSC_COMM_SELF, vecBelow );CHKERRQ(ierr);
	ierr = VecSetType( *vecBelow, VECSEQ );CHKERRQ(ierr);
	ierr = VecSetFromOptions( *vecBelow );CHKERRQ(ierr);
	ierr = VecSetSizes( *vecBelow, PETSC_DECIDE, NZvec );CHKERRQ(ierr);
	ierr = VecSet( *vecBelow, 0 );CHKERRQ(ierr);
	ierr = VecCreate( PETSC_COMM_SELF, vecAbove );CHKERRQ(ierr);
	ierr = VecSetType( *vecAbove, VECSEQ );CHKERRQ(ierr);
	ierr = VecSetFromOptions( *vecAbove );CHKERRQ(ierr);
	ierr = VecSetSizes( *vecAbove, PETSC_DECIDE, NZvec );CHKERRQ(ierr);
	ierr = VecSet( *vecAbove, 0 );CHKERRQ(ierr);

	ierr = MatDuplicate( *Q, MAT_COPY_VALUES, &halfQ );CHKERRQ(ierr);
	ierr = MatScale( halfQ, 0.5 );
	ierr = MatGetRow( halfQ, Ji-1, &nvals, &indices, &values );CHKERRQ(ierr);
	ierr = VecSetValues( *vecBelow, nvals, indices, values, INSERT_VALUES );CHKERRQ(ierr);
	ierr = VecSetValues( *vecBelow, 1, &diagIndBelow, &diag, ADD_VALUES );CHKERRQ(ierr);
	*nonzeros = nvals;
	ierr = MatRestoreRow( halfQ, Ji-1, &nvals, &indices, &values );CHKERRQ(ierr);

	ierr = MatGetRow( halfQ, Ji, &nvals, &indices, &values );CHKERRQ(ierr);
	ierr = VecSetValues( *vecAbove, nvals, indices, values, INSERT_VALUES );CHKERRQ(ierr);
	ierr = VecSetValues( *vecAbove, 1, &diagIndAbove, &diag, ADD_VALUES );CHKERRQ(ierr);
	*nonzeros = NCPA::max( nvals, *nonzeros );
	ierr = MatRestoreRow( halfQ, Ji, &nvals, &indices, &values );CHKERRQ(ierr);
	
	ierr = VecAssemblyBegin( *vecBelow );CHKERRQ(ierr);
	ierr = VecAssemblyEnd( *vecBelow );CHKERRQ(ierr);
	ierr = VecAssemblyBegin( *vecAbove );CHKERRQ(ierr);
	ierr = VecAssemblyEnd( *vecAbove );CHKERRQ(ierr);

	ierr = MatDestroy( &halfQ );CHKERRQ(ierr);
	*nonzeros = NZvec / 10;
	return 1;
	

	/* For order 1,1 and 2,2 */
	// std::vector< PetscScalar > numerator_coefficients, denominator_coefficients;
	// Vec ek_below, ek_above, Be_below, Be_above;
	// KSP ksp;
	// Mat B, C, Ctrans, *last_q_powers;
	
	// PetscInt ncoeffs = 3;
	// numerator_coefficients.push_back( 1.0 );
	// numerator_coefficients.push_back( 1.25 );
	// numerator_coefficients.push_back( 5.0/16.0 );
	// denominator_coefficients.push_back( 1.0 );
	// denominator_coefficients.push_back( 0.75 );
	// denominator_coefficients.push_back( 1.0/16.0 );

	// PetscInt ncoeffs = 2;
	// numerator_coefficients.push_back( 1.0 );
	// numerator_coefficients.push_back( 0.75 );
	// denominator_coefficients.push_back( 1.0 );
	// denominator_coefficients.push_back( 0.25 );

	// last_q_powers = NULL;
	// create_polymatrix_vector( ncoeffs, Q, &last_q_powers );
	// generate_polymatrices( last_q_powers, ncoeffs, NZvec, numerator_coefficients, denominator_coefficients, &B, &C );

	// // create index vectors
	// ierr = VecCreate( PETSC_COMM_SELF, &ek_below );CHKERRQ(ierr);
	// ierr = VecSetType( ek_below, VECSEQ );CHKERRQ(ierr);
	// ierr = VecSetSizes( ek_below, PETSC_DECIDE, NZvec );CHKERRQ(ierr);
	// ierr = VecSetFromOptions( ek_below );CHKERRQ(ierr);
	// ierr = VecSet( ek_below, 0 );CHKERRQ(ierr);
	// ierr = VecCreate( PETSC_COMM_SELF, &ek_above );CHKERRQ(ierr);
	// ierr = VecSetType( ek_above, VECSEQ );CHKERRQ(ierr);
	// ierr = VecSetSizes( ek_above, PETSC_DECIDE, NZvec );CHKERRQ(ierr);
	// ierr = VecSetFromOptions( ek_above );CHKERRQ(ierr);
	// ierr = VecSet( ek_above, 0 );CHKERRQ(ierr);

	// if (Ji > 0) {
	// 	ierr = VecSetValue( ek_below, Ji-1, 1, INSERT_VALUES );CHKERRQ(ierr);
	// }
	// ierr = VecSetValue( ek_above, Ji, 1, INSERT_VALUES );CHKERRQ(ierr);
	// ierr = VecAssemblyBegin( ek_above );CHKERRQ(ierr);
	// ierr = VecAssemblyEnd( ek_above );CHKERRQ(ierr);
	// ierr = VecAssemblyBegin( ek_below );CHKERRQ(ierr);
	// ierr = VecAssemblyEnd( ek_below );CHKERRQ(ierr);

	// ierr = VecCreate( PETSC_COMM_SELF, &Be_below );CHKERRQ(ierr);
	// ierr = VecSetType( Be_below, VECSEQ );CHKERRQ(ierr);
	// ierr = VecSetSizes( Be_below, PETSC_DECIDE, NZvec );CHKERRQ(ierr);
	// ierr = VecSetFromOptions( Be_below );CHKERRQ(ierr);
	// ierr = VecAssemblyBegin( Be_below );CHKERRQ(ierr);
	// ierr = VecAssemblyEnd( Be_below );CHKERRQ(ierr);

	// ierr = VecCreate( PETSC_COMM_SELF, &Be_above );CHKERRQ(ierr);
	// ierr = VecSetType( Be_above, VECSEQ );CHKERRQ(ierr);
	// ierr = VecSetSizes( Be_above, PETSC_DECIDE, NZvec );CHKERRQ(ierr);
	// ierr = VecSetFromOptions( Be_above );CHKERRQ(ierr);
	// ierr = VecAssemblyBegin( Be_above );CHKERRQ(ierr);
	// ierr = VecAssemblyEnd( Be_above );CHKERRQ(ierr);

	// // setup RHS and result vectors
	// //ierr = MatCreateTranspose( B, &Btrans );CHKERRQ(ierr);
	// ierr = MatCreateTranspose( C, &Ctrans );CHKERRQ(ierr);
	// ierr = MatMultTranspose( B, ek_below, Be_below );CHKERRQ(ierr);
	// ierr = MatMultTranspose( B, ek_above, Be_above );CHKERRQ(ierr);
	// ierr = VecCreate( PETSC_COMM_SELF, vecBelow );CHKERRQ(ierr);
	// ierr = VecSetType( *vecBelow, VECSEQ );CHKERRQ(ierr);
	// ierr = VecSetFromOptions( *vecBelow );CHKERRQ(ierr);
	// ierr = VecSetSizes( *vecBelow, PETSC_DECIDE, NZvec );CHKERRQ(ierr);
	// ierr = VecCreate( PETSC_COMM_SELF, vecAbove );CHKERRQ(ierr);
	// ierr = VecSetType( *vecAbove, VECSEQ );CHKERRQ(ierr);
	// ierr = VecSetFromOptions( *vecAbove );CHKERRQ(ierr);
	// ierr = VecSetSizes( *vecAbove, PETSC_DECIDE, NZvec );CHKERRQ(ierr);

	// // Set up solution
	// ierr = KSPCreate( PETSC_COMM_SELF, &ksp );CHKERRQ(ierr);
	// ierr = KSPSetOperators( ksp, Ctrans, Ctrans );CHKERRQ(ierr);
	// ierr = KSPSetFromOptions( ksp );CHKERRQ(ierr);
	// ierr = KSPSolve( ksp, Be_below, *vecBelow );CHKERRQ(ierr);
	// ierr = KSPSetOperators( ksp, Ctrans, Ctrans );CHKERRQ(ierr);
	// ierr = KSPSolve( ksp, Be_above, *vecAbove );CHKERRQ(ierr);

	// // clean up
	// ierr = KSPDestroy( &ksp );CHKERRQ(ierr);
	// ierr = VecDestroy( &ek_below );CHKERRQ(ierr);
	// ierr = VecDestroy( &ek_above );CHKERRQ(ierr);
	// ierr = VecDestroy( &Be_above );CHKERRQ(ierr);
	// ierr = VecDestroy( &Be_below );CHKERRQ(ierr);
	// // ierr = MatDestroy( &Btrans );CHKERRQ(ierr);
	// ierr = MatDestroy( &Ctrans );CHKERRQ(ierr);
	// ierr = MatDestroy( &B );CHKERRQ(ierr);
	// ierr = MatDestroy( &C );CHKERRQ(ierr);
	// delete_polymatrix_vector( ncoeffs, &last_q_powers );
	// *nonzeros = NZvec;
	// return 1;

}










int NCPA::EPadeSolver::make_q_powers( NCPA::Atmosphere2D *atm, int NZvec, double *zvec, 
	double r, std::complex<double> *k, double k0, double h2, double z_s,
	std::complex<double> impedence_factor, std::complex<double> *n, size_t nqp, 
	int boundary_index, const Mat &last_q, Mat *qpowers ) {

	Mat q;
	PetscInt Istart, Iend, col[3];
	PetscBool FirstBlock = PETSC_FALSE, LastBlock = PETSC_FALSE;
	PetscErrorCode ierr;
	PetscScalar value[3];
	PetscInt i;

	// Calculate parameters
	double h = std::sqrt( h2 );
	double J_s = (z_s - zvec[0]) / h;
	int Ji = NCPA::find_closest_index( zvec, NZvec, z_s );  // first index above ground
	if ( zvec[ Ji ] < z_s ) {
		Ji++;
	}
	double dJ = (double)Ji;
	//double dJ = zvec[ Ji ] / h;

	// calculate intermediate variables as shown in notes
	double rho_a = atm->get( r, "RHO", z_s );
	double Gamma = 0.5 * atm->get_first_derivative( r, "RHO", z_s ) / rho_a;
	double rho_b = rho_a * 1000.0;
	double Anom = (1.0 / rho_a) * (1.0 / (dJ - J_s));
	double Bnom = (1.0 / rho_b) * (1.0 / (J_s - dJ + 1.0));
	double denom = Anom + Bnom - (h * Gamma);
	double s_A = Anom / denom;
	double s_B = Bnom / denom;
	double a, b, c, alpha, beta, gamma;
	if (last_q == PETSC_NULL) {
		a   = s_B / (dJ - J_s);
		b   = (s_A - 2.0) / (dJ - J_s);
		c   = 1.0 / (dJ - J_s);
		alpha = 1.0 / (J_s - dJ + 1.0);
		beta  = (s_B - 2.0) / (J_s - dJ + 1.0);
		gamma = s_A / (J_s - dJ + 1.0);
	} else {
		PetscScalar I( 0.0, 1.0 );
		PetscScalar M = I * k0 * atm->get_interpolated_ground_elevation_first_derivative( r ) * h / denom;
		std::vector< PetscScalar > numerator_coefficients, denominator_coefficients;
		numerator_coefficients.push_back( 1.0 );
		numerator_coefficients.push_back( 1.25 );
		numerator_coefficients.push_back( 5.0/16.0 );
		denominator_coefficients.push_back( 1.0 );
		denominator_coefficients.push_back( 0.75 );
		denominator_coefficients.push_back( 1.0/16.0 );
		
	}


	// Set up matrices
	ierr = MatCreateSeqAIJ( PETSC_COMM_SELF, NZvec, NZvec, 3, NULL, &q );CHKERRQ(ierr);
	ierr = MatSetFromOptions( q );CHKERRQ(ierr);

	// populate
	//double bnd_cnd = -1.0 / h2;    // @todo add hook for alternate boundary conditions
	//double bnd_cnd = -2.0 / h2;      // pressure release condition
	//std::complex<double> bnd_cnd = (impedence_factor * std::sqrt( h2 ) - 1.0) / h2;
	double k02 = k0*k0;
	
	// If this process is being split over processors, we need to check to see
	// if this particular instance contains the first or last rows, because those
	// get filled differently
	ierr = MatGetOwnershipRange(q,&Istart,&Iend);CHKERRQ(ierr);

	// Does this instance contain the first row?
	if (Istart == 0) {
		FirstBlock=PETSC_TRUE;
	}

	// Does this instance contain the last row?
    if (Iend==NZ) {
    	LastBlock=PETSC_TRUE;
    }

    //value[0]=1.0 / h2 / k02; value[2]=1.0 / h2 / k02;
    // iterate over block.  If this instance contains the first row, leave that one
    // for later, same for if this instance contains the last row.
    double Drow[ 3 ];
    memset( Drow, 0, 3*sizeof(double) );
    for( i=(FirstBlock? Istart+1: Istart); i<(LastBlock? Iend-1: Iend); i++ ) {

		// set column numbers.  Since the matrix Q is tridiagonal (because input 
		// matrix D is tridiagonal and K is diagonal), column indices are
		// i-1, i, i+1
    	col[ 0 ] = i-1;
    	col[ 1 ] = i;
    	col[ 2 ] = i+1;

    	// Set values.  This will be the same unless we're at the indices immediately
    	// below or above the ground surface
    	if (i == (Ji-1)) {
    		// this is the alpha, beta, gamma row
    		Drow[0] = alpha;
    		Drow[1] = beta;
    		Drow[2] = gamma;
    	} else if (i == Ji) {
    		// this is the a, b, c row
    		Drow[0] = a;
    		Drow[1] = b;
    		Drow[2] = c;
    	} else {
    		Drow[0] = 1.0;
    		Drow[1] = -2.0;
    		Drow[2] = 1.0;
    	}
    	value[ 0 ] = Drow[0] / h2 / k02;
    	value[ 1 ] = ( (Drow[1] / h2) + k[ i ]*k[ i ] - k02 ) / k02;
    	value[ 2 ] = Drow[2] / h2 / k02;
		ierr = MatSetValues(q,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (LastBlock) {
		    i=NZ-1; col[0]=NZ-2; col[1]=NZ-1;
		    value[ 0 ] = 1.0 / h2 / k02;
		    //value[ 1 ] = -2.0/h2/k02 + (n[i]*n[i] - 1);
		    value[ 1 ] = ( (-2.0 / h2) + k[ i ]*k[ i ] - k02 ) / k02;
		    ierr = MatSetValues(q,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (FirstBlock) {
		    i=0; col[0]=0; col[1]=1; 
		    if (i == (Ji-1))  {
		    	Drow[0] = alpha;
	    		Drow[1] = beta;
	    		Drow[2] = gamma;
    			// value[ 0 ] = 0.0;
    			// value[ 1 ] = 0.0;
    		} else if (i == Ji) {
    			Drow[0] = a;
	    		Drow[1] = b;
	    		Drow[2] = c;
    		} else {
    			Drow[0] = 1.0;
	    		Drow[1] = -2.0;
	    		Drow[2] = 1.0;
    		}
    		value[ 0 ] = ( (Drow[1] / h2) + k[ i ]*k[ i ] - k02 ) / k02;
    		value[ 1 ] = Drow[2] / h2 / k02;
		    ierr = MatSetValues(q,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    // calculate powers of q
	//qpowers = new Mat[ nqp ];
	ierr = MatConvert( q, MATSAME, MAT_INITIAL_MATRIX, qpowers );CHKERRQ(ierr);
	for (i = 1; i < (PetscInt)nqp; i++) {
		ierr = MatMatMult( qpowers[i-1], qpowers[0], MAT_INITIAL_MATRIX, PETSC_DEFAULT, 
			qpowers+i );CHKERRQ(ierr);
	}
	ierr = MatDestroy( &q );CHKERRQ(ierr);
	return 1;
}

void NCPA::EPadeSolver::absorption_layer( double lambda, double *z, int NZ, double *layer ) {
	double z_t = z[NZ-1] - lambda;
	for (int i = 0; i < NZ; i++) {
		layer[ i ] = absorption_layer_mu * std::exp( (z[i]-z_t) / lambda );
	}
}


int NCPA::EPadeSolver::get_starter_gaussian( size_t NZ, double *z, double zs, double k0, int ground_index,
	Vec *psi ) {

	double fac = 2.0;
	//double kfac = k0 / fac;
	PetscScalar tempval;
	PetscErrorCode ierr;

	ierr = VecCreate( PETSC_COMM_SELF, psi );CHKERRQ(ierr);
	ierr = VecSetSizes( *psi, PETSC_DECIDE, NZ );CHKERRQ(ierr);
	ierr = VecSetFromOptions( *psi ); CHKERRQ(ierr);
	ierr = VecSet( *psi, 0.0 );

	for (PetscInt i = 0; i < (PetscInt)NZ; i++) {
		//if (z[i] >= zg) {
			tempval = -( k0*k0/fac/fac ) * (z[i] - zs) * (z[i] - zs);
			tempval = sqrt( k0/fac ) * exp( tempval );
			ierr = VecSetValues( *psi, 1, &i, &tempval, INSERT_VALUES );CHKERRQ(ierr);
		//}
	}
	ierr = VecAssemblyBegin( *psi );CHKERRQ(ierr);
	ierr = VecAssemblyEnd( *psi );CHKERRQ(ierr);
	return 1;
}

int NCPA::EPadeSolver::get_starter_self_revised( size_t NZ, double *z, double zs, 
	double r_ref, double z_ground, double k0, Mat *qpowers, size_t npade, Vec *psi ) {

	Mat B, C, A, BtAt;
	KSP ksp;
	PetscErrorCode ierr;
	Vec del, rhs;
	PetscScalar J( 0.0, 1.0 );

	// compute first pade approximation
	std::vector< PetscScalar > P, Q;
	std::vector< PetscScalar > taylor1 = taylor_exp_id_sqrt_1pQ_m1( 2*npade, k0*r_ref );
	calculate_pade_coefficients( &taylor1, npade, npade+1, &P, &Q );

	std::cout << "k0 = " << k0 << std::endl;
	printVector( "Taylor Coefficients for Exponential", taylor1 );
	printVector( "Numerator Pade Coefficients", P );
	printVector( "Denominator Pade Coefficients", Q );
	generate_polymatrices( qpowers, npade, NZ, P, Q, &B, &C );

	// compute second pade approximation
	std::vector< PetscScalar > taylor2 = taylor_1pQ_n025( 2*npade );
	printVector( "Taylor Coefficients for Quarter Root", taylor2 );
	generate_polymatrix( qpowers, npade+1, NZ, taylor2, &A );
	outputQ( "q_starter.dat", qpowers, NZ );

	// solve the equation:
	// Ct Pt = (2*pi*i/(k_0*r)) (Bt*At) delta_j,j0
	// First, intermediate matrix products
	ierr = MatTranspose( A, MAT_INPLACE_MATRIX, &A );CHKERRQ(ierr);
	ierr = MatTranspose( B, MAT_INPLACE_MATRIX, &B );CHKERRQ(ierr);
	ierr = MatTranspose( C, MAT_INPLACE_MATRIX, &C );CHKERRQ(ierr);
	ierr = MatMatMult( B, A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &BtAt );CHKERRQ(ierr);

	// Scale the right side matrix product by (2*PI*J)/(k0*r)
	// The Pade approximation above has an extra factor of exp( -J*k0*r ) so we 
	// take that out here a well
	PetscScalar mod = std::exp( J * k0 * r_ref );  
	ierr = MatScale( BtAt, std::sqrt( 2 * PI * J / k0 / r_ref ) * mod );CHKERRQ(ierr);

	// Find the closest index to the source height
	PetscInt nzsrc = (PetscInt)find_closest_index( z, NZ, zs );
	while (z[nzsrc] < z_ground) {
		nzsrc++;
	}

	// Set up delta function vector
	ierr = VecCreate( PETSC_COMM_SELF, &del );CHKERRQ(ierr);
	ierr = VecSetFromOptions( del );CHKERRQ(ierr);
	ierr = VecSetSizes( del, PETSC_DECIDE, NZ );CHKERRQ(ierr);
	ierr = VecSet( del, 0 );CHKERRQ(ierr);
	ierr = VecSetValue( del, nzsrc, 1, INSERT_VALUES );CHKERRQ(ierr);
	ierr = VecAssemblyBegin( del );CHKERRQ(ierr);
	ierr = VecAssemblyEnd( del );CHKERRQ(ierr);

	// Set up RHS vector = Bt * At * delta
	ierr = VecCreate( PETSC_COMM_SELF, &rhs );CHKERRQ(ierr);
	ierr = VecSetFromOptions( rhs );CHKERRQ(ierr);
	ierr = VecSetSizes( rhs, PETSC_DECIDE, NZ );CHKERRQ(ierr);
	ierr = VecAssemblyBegin( rhs );CHKERRQ(ierr);
	ierr = VecAssemblyEnd( rhs );CHKERRQ(ierr);
	ierr = MatMult( BtAt, del, rhs );CHKERRQ(ierr);

	// Set up lefthand side vector to solve for
	ierr = VecCreate( PETSC_COMM_WORLD, psi );CHKERRQ(ierr);
	ierr = VecSetFromOptions( *psi );CHKERRQ(ierr);
	ierr = VecSetSizes( *psi, PETSC_DECIDE, NZ );CHKERRQ(ierr);
	ierr = VecAssemblyBegin( *psi );CHKERRQ(ierr);
	ierr = VecAssemblyEnd( *psi );CHKERRQ(ierr);

	// Solve Ct * psi = Bt * At * delta for psi
	ierr = KSPCreate( PETSC_COMM_WORLD, &ksp );CHKERRQ(ierr);
	ierr = KSPSetOperators( ksp, C, C );CHKERRQ(ierr);
	ierr = KSPSetFromOptions( ksp );CHKERRQ(ierr);
	ierr = KSPSolve( ksp, rhs, *psi );CHKERRQ(ierr);

	// clean up
	KSPDestroy( &ksp );
	VecDestroy( &rhs );
	VecDestroy( &del );
	MatDestroy( &B );
	MatDestroy( &C );
	MatDestroy( &A );
	MatDestroy( &BtAt );

	return 1;

	// intermediate matrix products
	// ierr = MatMatMult( D, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M );CHKERRQ(ierr);
	// ierr = MatMatMult( E, C, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &N );CHKERRQ(ierr);

	// extract row representing source height.  Make sure it's above the ground surface.
	
	// const PetscScalar *row;
	// ierr = MatGetRow( A, nzsrc, NULL, NULL, &row );CHKERRQ(ierr);
	// ierr = VecCreate( PETSC_COMM_SELF, psi );CHKERRQ(ierr);
	// ierr = VecSetSizes( *psi, PETSC_DECIDE, NZ );CHKERRQ(ierr);
	// ierr = VecSetFromOptions( *psi );CHKERRQ(ierr);
	// PetscInt *indices = new PetscInt[ NZ ];
	// for (unsigned int ii = 0; ii < NZ; ii++) {
	// 	indices[ ii ] = ii;
	// }
	// ierr = VecSetValues( *psi, NZ, indices, row, INSERT_VALUES );CHKERRQ(ierr);
	// ierr = VecAssemblyBegin( *psi );CHKERRQ(ierr);
	// ierr = VecAssemblyEnd( *psi );CHKERRQ(ierr);
	// ierr = MatRestoreRow( A, nzsrc, NULL, NULL, &row );CHKERRQ(ierr);

	// // Scale the row vector
	// PetscScalar J( 0.0, 1.0 );
	// //double r_ref = 2 * PI / k0;
	// double r_ref = 1.0;
	// PetscScalar s = std::sqrt( 2 * PI * J / k0 / r_ref ) * std::exp( J * k0 * r_ref );  // r == 1
	// ierr = VecScale( *psi, s );CHKERRQ(ierr);
	// ierr = VecDuplicate( tempvec, psi );

	// // set up linear system
	// ierr = KSPCreate( PETSC_COMM_WORLD, &ksp );CHKERRQ(ierr);
	// ierr = KSPSetOperators( ksp, N, N );CHKERRQ(ierr);
	// ierr = KSPSetFromOptions( ksp );CHKERRQ(ierr);
	// ierr = KSPSolve( ksp, tempvec, *psi );CHKERRQ(ierr);

	// PetscScalar hank_inv = pow( sqrt( 2.0 / ( PI * k0 * r_ref ) ) * exp( J * (k0 * r_ref - PI/4.0 ) ),
	// 	-1.0 );
	// ierr = VecScale( *psi, hank_inv );
}

// int NCPA::EPadeSolver::get_starter_self( size_t NZ, double *z, double zs, double z_ground, 
// 	double k0, Mat *qpowers, size_t npade, Vec *psi ) {

// 	Vec rhs, ksi, Bksi, tempvec;
// 	Mat A, AA, B, C;
// 	KSP ksp, ksp2;
// 	PetscScalar I( 0.0, 1.0 ), tempsc, zeroval = 0.0;
// 	PetscInt ii, Istart, Iend;
// 	PetscErrorCode ierr;
	
// 	// create rhs vector
// 	ierr = VecCreate( PETSC_COMM_SELF, &rhs );CHKERRQ(ierr);
// 	ierr = VecSetSizes( rhs, PETSC_DECIDE, NZ );CHKERRQ(ierr);
// 	ierr = VecSetFromOptions( rhs );CHKERRQ(ierr);
// 	ierr = VecSet( rhs, 0.0 );CHKERRQ(ierr);
	
// 	// find closest index to zs. Make sure the picked point is above the ground surface
// 	PetscInt nzsrc = (PetscInt)find_closest_index( z, NZ, zs );
// 	while (z[nzsrc] < z_ground) {
// 		nzsrc++;
// 	}

// 	double h = z[1] - z[0];
// 	PetscScalar hinv = 1.0 / h;
// 	ierr = VecSetValues( rhs, 1, &nzsrc, &hinv, INSERT_VALUES );CHKERRQ(ierr);
	
// 	// set up identity matrix.  If this works, use it elsewhere
// 	ierr = MatCreateSeqAIJ( PETSC_COMM_SELF, NZ, NZ, 3, NULL, &A );CHKERRQ(ierr);
// 	ierr = MatSetFromOptions( A );CHKERRQ(ierr);
// 	ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
// 	tempsc = 1.0;
//     for (ii = Istart; ii < Iend; ii++) {
//     	ierr = MatSetValues( A, 1, &ii, 1, &ii, &tempsc, INSERT_VALUES);CHKERRQ(ierr);
//     }
//     ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
//     ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
//     ierr = MatAXPY( A, -I, qpowers[0], DIFFERENT_NONZERO_PATTERN );CHKERRQ(ierr);

// 	// square
// 	ierr = MatMatMult( A, A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &AA );
	
// 	// solve first part (Eq. 26)
// 	ierr = VecDuplicate( rhs, &ksi );CHKERRQ(ierr);
// 	ierr = VecSet( ksi, 0.0 );CHKERRQ(ierr);
// 	ierr = KSPCreate( PETSC_COMM_WORLD, &ksp );CHKERRQ(ierr);
// 	ierr = KSPSetOperators( ksp, AA, AA );CHKERRQ(ierr);
// 	ierr = KSPSetFromOptions( ksp );CHKERRQ(ierr);
// 	ierr = KSPSolve( ksp, rhs, ksi );CHKERRQ(ierr);
	
// 	// get starter
// 	std::cout << "Finding ePade starter coefficients..." << std::endl;
// 	double r_ref = 2 * PI / k0;
// 	std::vector<PetscScalar> P, Q;
// 	epade( npade, k0, r_ref, &P, &Q, true );

// 	ierr = MatCreateSeqAIJ( PETSC_COMM_SELF, NZ, NZ, 2*npade-1, NULL, &B );CHKERRQ(ierr);
// 	ierr = MatSetFromOptions( B );CHKERRQ(ierr);
// 	ierr = MatCreateSeqAIJ( PETSC_COMM_SELF, NZ, NZ, 2*npade+1, NULL, &C );CHKERRQ(ierr);
// 	ierr = MatSetFromOptions( C );CHKERRQ(ierr);

// 	ierr = MatGetOwnershipRange(B,&Istart,&Iend);CHKERRQ(ierr);
// 	PetscScalar value = 1.0;
// 	for (ii = Istart; ii < Iend; ii++) {
// 		ierr = MatSetValues( B, 1, &ii, 1, &ii, &value, INSERT_VALUES );CHKERRQ(ierr);
// 		ierr = MatSetValues( C, 1, &ii, 1, &ii, &value, INSERT_VALUES );CHKERRQ(ierr);
// 	}

// 	ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
//     ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
//     ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
//     ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

//     for (ii = 1; ii < (PetscInt)(Q.size()); ii++) {
// 		ierr = MatAXPY( C, Q[ ii ], qpowers[ ii-1 ], DIFFERENT_NONZERO_PATTERN );CHKERRQ(ierr);
// 	}
// 	for (ii = 1; ii < (PetscInt)(P.size()); ii++) {
// 		ierr = MatAXPY( B, P[ ii ], qpowers[ ii-1 ], DIFFERENT_NONZERO_PATTERN );CHKERRQ(ierr);
// 	}

// 	PetscScalar hank_inv = pow( sqrt( 2.0 / ( PI * k0 * r_ref ) ) * exp( I * (k0 * r_ref - PI/4.0 ) ),
// 		-1.0 );

// 	// Original Matlab: psi = AA * ( C \ (B * ksi) ) / hank
// 	// compute product of B and ksi
// 	ierr = VecDuplicate( ksi, &Bksi );
// 	ierr = VecDuplicate( ksi, &tempvec );
// 	ierr = VecDuplicate( ksi, psi );
// 	ierr = MatMult( B, ksi, Bksi );

// 	// solve for tempvec = C \ Bksi
// 	ierr = KSPCreate( PETSC_COMM_WORLD, &ksp2 );CHKERRQ(ierr);
// 	ierr = KSPSetOperators( ksp2, C, C );CHKERRQ(ierr);
// 	ierr = KSPSetFromOptions( ksp2 );CHKERRQ(ierr);
// 	ierr = KSPSolve( ksp2, Bksi, tempvec );CHKERRQ(ierr);
	
// 	// multiply and scale
// 	ierr = MatMult( AA, tempvec, *psi );CHKERRQ(ierr);
// 	ierr = VecScale( *psi, hank_inv );CHKERRQ(ierr);


// 	// clean up
// 	ierr = MatDestroy( &A );CHKERRQ(ierr);
// 	ierr = MatDestroy( &AA );CHKERRQ(ierr);
// 	ierr = MatDestroy( &B );CHKERRQ(ierr);
// 	ierr = MatDestroy( &C );CHKERRQ(ierr);
// 	ierr = VecDestroy( &rhs );CHKERRQ(ierr);
// 	ierr = VecDestroy( &ksi );CHKERRQ(ierr);
// 	ierr = VecDestroy( &Bksi );CHKERRQ(ierr);
// 	ierr = VecDestroy( &tempvec );CHKERRQ(ierr);
// 	ierr = KSPDestroy( &ksp );CHKERRQ(ierr);
// 	ierr = KSPDestroy( &ksp2 );CHKERRQ(ierr);

// 	return 1;
// }






int NCPA::EPadeSolver::calculate_pade_coefficients( std::vector<PetscScalar> *c, 
	int n_numerator, int n_denominator, std::vector<PetscScalar> *numerator_coefficients,
	std::vector<PetscScalar> *denominator_coefficients ) {

	// sanity checks
	if (n_denominator < n_numerator) {
		std::cerr << "Denominator count must be >= numerator count for Pade calculation" << std::endl;
		exit(0);
	}
	int n = n_numerator - 1;    // numerator order
	int m = n_denominator - 1;  // denominator order
	int N = n + m;
	int n_taylor = c->size();
	if (n_taylor < (N+1)) {
		std::cerr << "Count of Taylor series must be at least " << (N+1) << " for numerator count "
				  << n_numerator << " and denominator count " << n_denominator << std::endl;
		exit(0);
	}

	//double delta = k0 * dr;
	std::complex<double> j( 0.0, 1.0 );
	PetscErrorCode ierr;
	PetscInt Istart, Iend, ii, jj, *indices;
	// PetscBool      FirstBlock=PETSC_FALSE, LastBlock=PETSC_FALSE;
	PetscScalar tempsc, *contents;
	Mat A;
	Vec x, y;
	KSP ksp;

	// Create and populate matrix system
	ierr = MatCreateSeqAIJ( PETSC_COMM_SELF, N, N, n_denominator, NULL, &A );CHKERRQ(ierr);
	ierr = MatSetFromOptions( A );CHKERRQ(ierr);
	ierr = MatZeroEntries( A );CHKERRQ(ierr);
	ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
	// if (Istart==0) FirstBlock=PETSC_TRUE;
 //    if (Iend==(N-1)) LastBlock=PETSC_TRUE;
    tempsc = -1.0;
    for (ii = Istart; ii < min(n,Iend); ii++) {
    	// std::cout << "A[ " << ii << "," << ii << " ] = " << tempsc << std::endl;
    	ierr = MatSetValues(A,1,&ii,1,&ii,&tempsc,INSERT_VALUES);CHKERRQ(ierr);
    }
    for (ii = Istart; ii < Iend; ii++) {
    	for (jj = n; jj <= min(Iend-1,ii+n); jj++) {
    		tempsc = c->at( ii-jj+n );
    		// std::cout << "A[ " << ii << "," << jj << " ] = " << tempsc << std::endl;
    		ierr = MatSetValues(A,1,&ii,1,&jj,&tempsc,INSERT_VALUES);CHKERRQ(ierr);
    	} 
    }


    // for (ii = Istart; ii < Iend; ii++) {
    // 	ierr = MatSetValues( A, 1, &ii, 1, &ii, &tempsc, INSERT_VALUES);CHKERRQ(ierr);
    // }
    // for (ii = Istart; ii < min(pM,Iend-1); ii++) {
    // 	for (jj = ii; jj < (N-1); jj++) {
    // 		ierr = MatSetValues(A,1,&jj,1,&ii,c+(jj-ii),INSERT_VALUES);CHKERRQ(ierr);
    // 	}
    // }
    // tempsc = -1.0;
    // for (ii = Istart; ii < min(pL,Iend); ii++) {
    // 	jj = ii+pM;
    // 	ierr = MatSetValues(A,1,&ii,1,&jj,&tempsc,INSERT_VALUES);CHKERRQ(ierr);
    // }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

	// setup right-side vector
	ierr = VecCreate( PETSC_COMM_SELF, &x );CHKERRQ(ierr);
	ierr = VecSetSizes( x, PETSC_DECIDE, N );CHKERRQ(ierr);
	ierr = VecSetFromOptions( x ); CHKERRQ(ierr);
	ierr = VecDuplicate( x, &y );CHKERRQ(ierr);

	//indices = new PetscInt[ N-1 ];
	indices = new PetscInt[ N ];
	// for (ii = 0; ii < (N-1); ii++) {
	for (ii = 0; ii < N; ii++) {
		tempsc = -c->at( ii+1 );
		// std::cout << "y[ " << ii << " ] = -c[ " << ii+1 << " ] = " << tempsc << std::endl;
		ierr = VecSetValues( y, 1, &ii, &tempsc, INSERT_VALUES );CHKERRQ(ierr);
		indices[ ii ] = ii;
	}
	ierr = VecAssemblyBegin( y );CHKERRQ(ierr);
	ierr = VecAssemblyEnd( y );CHKERRQ(ierr);

	ierr = VecSet( x, 0.0 );CHKERRQ(ierr);

	// solve
	ierr = KSPCreate( PETSC_COMM_WORLD, &ksp );CHKERRQ(ierr);
	ierr = KSPSetOperators( ksp, A, A );CHKERRQ(ierr);
	ierr = KSPSetFromOptions( ksp );CHKERRQ(ierr);
	ierr = KSPSolve( ksp, y, x );CHKERRQ(ierr);

	// populate P and Q vectors. Q is denominator coefficients (b), P is numerator coefficients (a)
	contents = new PetscScalar[ N ];
	ierr = VecGetValues( x, N, indices, contents );

	numerator_coefficients->clear();
	numerator_coefficients->push_back( c->at(0) );
	for (ii = 0; ii < n; ii++) {
		numerator_coefficients->push_back( contents[ ii ] );
	}
	denominator_coefficients->clear();
	denominator_coefficients->push_back( 1.0 );
	for (ii = n; ii < N; ii++) {
		denominator_coefficients->push_back( contents[ ii ] );
	}
	delete [] contents;
	delete [] indices;
	
	// clean up memory
	ierr = KSPDestroy( &ksp );CHKERRQ(ierr);
	ierr = VecDestroy( &x );CHKERRQ(ierr);
	ierr = VecDestroy( &y );CHKERRQ(ierr);
	ierr = MatDestroy( &A );CHKERRQ(ierr);
	return 0;
}

// Uses the recurrence relation in Roberts & Thompson (2013, eq. 16) to compute the
// Taylor series coefficients for F=exp[ i*d*( sqrt(1+Q) - 1 ) ] up to order N-1 
// (i.e. the first N terms)
std::vector<PetscScalar> NCPA::EPadeSolver::taylor_exp_id_sqrt_1pQ_m1( int N, double delta ) {
	std::complex<double> j( 0.0, 1.0 );
	std::vector<PetscScalar> c( N, 1.0 );
	//std::memset( c, 0, N*sizeof(PetscScalar) );
	//c[ 0 ] = 1.0;
	c[ 1 ] = j * 0.5 * delta;
	for ( int idx= 2; idx < N; idx++) {
		double dm = (double)(idx - 1);
		c[ idx ] = -((2.0*dm - 1.0) / (2.0*dm + 2.0)) * c[idx-1]
				   - (delta*delta / (4.0*dm*(dm+1.0))) * c[idx-2];
	}

	return c;
}

// Uses a calculated recursion relation to compute the Taylor series coefficients for
// G=(1+Q)^-0.25 up to order N-1 (i.e. the first N terms)
std::vector<PetscScalar> NCPA::EPadeSolver::taylor_1pQ_n025( int N ) {
	std::vector<PetscScalar> c( N, 1.0 );
	c[ 1 ] = -0.25;
	for (int idx = 2; idx < N; idx++) {
		double dn = double(idx);
		c[ idx ] = -((4*dn)-3) * c[ idx-1 ] / (4*dn); 
	}

	return c;
}

// Uses a calculated recursion relation to compute the Taylor series coefficients for
// G=(1+Q)^0.25 up to order N-1 (i.e. the first N terms)
std::vector<PetscScalar> NCPA::EPadeSolver::taylor_1pQ_025( int N ) {
	std::vector<PetscScalar> c( N, 1.0 );
	c[ 1 ] = 0.25;
	for (int idx = 1; idx < N; idx++) {
		double dn = double(idx);
		c[ idx ] = c[ idx-1 ] * (1.0/dn) * -((4.0*dn - 5.0)/4.0);
	}

	return c;
}

std::vector<PetscScalar> NCPA::EPadeSolver::taylor_1pQpid_n025( int N, double delta ) {
	std::vector<PetscScalar> c( N, 0.0 );
	std::complex< double > J( 0.0, 1.0 );
	c[ 0 ] = std::pow( 1.0 + J*delta, -0.25 );
	for (int idx = 1; idx < N; idx++) {
		double dn = double(idx);
		c[ idx ] = c[ idx-1 ] * (1.0/dn) * (-(4.0*(dn-1.0)+1.0) / 4) * std::pow( 1.0 + J*delta, -1.0 );
	}

	return c;
}


// int NCPA::EPadeSolver::epade( int order, double k0, double dr, std::vector<PetscScalar> *P, 
// 		std::vector<PetscScalar> *Q, bool starter ) {

// 	int M = order, N = 2*order;
// 	int L = N - 1 - M;
// 	std::complex<double> j( 0.0, 1.0 );
// 	double delta = k0 * dr;
// 	int m;

// 	PetscErrorCode ierr;
// 	PetscInt Istart, Iend, ii, jj, pM = M, pL = L, *indices;
// 	PetscBool      FirstBlock=PETSC_FALSE, LastBlock=PETSC_FALSE;
// 	PetscScalar tempsc, *contents;
// 	Mat A;
// 	Vec x, y;
// 	KSP ksp;
// 	//PC pc;

// 	// Create the temporary coefficient vector c
// 	// using complex<double> to make use of the built-in calculations
// 	P->clear();
// 	Q->clear();

// 	// From Roberts & Thompson, 2013, Eq. 16
// 	std::complex<double> *c = new std::complex<double>[ N ];
// 	std::memset( c, 0, N*sizeof(std::complex<double>));
// 	c[ 0 ] = 1.0;
// 	c[ 1 ] = j * 0.5 * delta;
// 	for (m = 1; m <= (2*M - 2); m++) {
// 		int idx = m+1;
// 		double dm = (double)m;
// 		c[ idx ] = -((2.0*dm - 1.0) / (2.0*dm + 2.0)) * c[idx-1]
// 				   - (delta*delta / (4.0*dm*(dm+1.0))) * c[idx-2];
// 	}

// 	// apply modification for starter calculation
// 	if (starter) {
// 		std::complex<double> *d = new std::complex<double>[ N ];
// 		std::memset( d, 0, N*sizeof(std::complex<double>) );
// 		d[0] = 1.0;
// 		for (m = 1; m <= (2*M - 1); m++) {
// 			d[ m ] = (j*delta*c[m-1] - (((double)(1 + 2*(m-1))) * d[m-1])) / (2.0*m);
// 		}
// 		std::memcpy( c, d, N*sizeof(std::complex<double>) );
// 		delete [] d;
// 	}

// 	// Create and populate matrix system A
// 	ierr = MatCreateSeqAIJ( PETSC_COMM_SELF, N-1, N-1, M+1, NULL, &A );CHKERRQ(ierr);
// 	ierr = MatSetFromOptions( A );CHKERRQ(ierr);
// 	ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
// 	if (Istart==0) FirstBlock=PETSC_TRUE;
//     if (Iend==(N-1)) LastBlock=PETSC_TRUE;
//     tempsc = 0.0;
//     for (ii = Istart; ii < Iend; ii++) {
//     	ierr = MatSetValues( A, 1, &ii, 1, &ii, &tempsc, INSERT_VALUES);CHKERRQ(ierr);
//     }
//     for (ii = Istart; ii < min(pM,Iend-1); ii++) {
//     	for (jj = ii; jj < (N-1); jj++) {
//     		ierr = MatSetValues(A,1,&jj,1,&ii,c+(jj-ii),INSERT_VALUES);CHKERRQ(ierr);
//     	}
//     }
//     tempsc = -1.0;
//     for (ii = Istart; ii < min(pL,Iend); ii++) {
//     	jj = ii+pM;
//     	ierr = MatSetValues(A,1,&ii,1,&jj,&tempsc,INSERT_VALUES);CHKERRQ(ierr);
//     }
//     ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
//     ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

// 	// setup right-side vector
// 	ierr = VecCreate( PETSC_COMM_SELF, &x );CHKERRQ(ierr);
// 	ierr = VecSetSizes( x, PETSC_DECIDE, N-1 );CHKERRQ(ierr);
// 	ierr = VecSetFromOptions( x ); CHKERRQ(ierr);
// 	ierr = VecDuplicate( x, &y );CHKERRQ(ierr);

// 	indices = new PetscInt[ N-1 ];
// 	for (ii = 0; ii < (N-1); ii++) {
// 		tempsc = -c[ ii+1 ];
// 		ierr = VecSetValues( y, 1, &ii, &tempsc, INSERT_VALUES );CHKERRQ(ierr);
// 		indices[ ii ] = ii;
// 	}
// 	ierr = VecAssemblyBegin( y );CHKERRQ(ierr);
// 	ierr = VecAssemblyEnd( y );CHKERRQ(ierr);

// 	ierr = VecSet( x, 0.0 );CHKERRQ(ierr);

// 	// solve
// 	ierr = KSPCreate( PETSC_COMM_WORLD, &ksp );CHKERRQ(ierr);
// 	ierr = KSPSetOperators( ksp, A, A );CHKERRQ(ierr);
// 	ierr = KSPSetFromOptions( ksp );CHKERRQ(ierr);
// 	ierr = KSPSolve( ksp, y, x );CHKERRQ(ierr);

// 	// populate P and Q vectors 
// 	Q->push_back( 1.0 );
// 	contents = new PetscScalar[ N-1 ];
// 	ierr = VecGetValues( x, N-1, indices, contents );

// 	for (ii = 0; ii < M; ii++) {
// 		Q->push_back( contents[ ii ] );
// 	}
// 	P->push_back( c[ 0 ] );
// 	for (ii = M; ii < (M+L); ii++) {
// 		P->push_back( contents[ ii ] );
// 	}
// 	delete [] contents;
// 	delete [] indices;
// 	delete [] c;
	
// 	// clean up memory
// 	ierr = KSPDestroy( &ksp );CHKERRQ(ierr);
// 	ierr = VecDestroy( &x );CHKERRQ(ierr);
// 	ierr = VecDestroy( &y );CHKERRQ(ierr);
// 	ierr = MatDestroy( &A );CHKERRQ(ierr);
// 	return 0;
// }

void NCPA::EPadeSolver::output1DTL( std::string filename, bool append ) {
	std::ofstream out_1d;
	if (append) {
		out_1d.open( filename, std::ofstream::out | std::ofstream::app );
		out_1d << std::endl;
	} else {
		out_1d.open( filename, std::ofstream::out | std::ofstream::trunc );
	}
	for (int i = 0; i < (NR-1); i++) {
		out_1d << r[ i ]/1000.0 << " " << calc_az << " " << tl[ zgi_r[ i ] ][ i ].real()
		       << " " << tl[ zgi_r[ i ] ][ i ].imag() << std::endl;
	}
	out_1d.close();
}

void NCPA::EPadeSolver::output2DTL( std::string filename ) {
	std::ofstream out_2d( filename, std::ofstream::out | std::ofstream::trunc );
	int zplot_int = 10;
	for (int i = 0; i < (NR-1); i++) {
		for (int j = 0; j < NZ; j += zplot_int) {
			//out_2d << r[ i ]/1000.0 << " " << z[ j ]/1000.0 << " " << tl[ j ][ i ] << " 0.0" << std::endl;
			out_2d << r[ i ]/1000.0 << " " << z[ j ]/1000.0 << " " << tl[ j ][ i ].real() 
			<< " " << tl[ j ][ i ].imag() << std::endl;
		}
		out_2d << std::endl;
	}
	out_2d.close();
}

void NCPA::EPadeSolver::set_1d_output( bool tf ) {
	write1d = tf;
}

/*
Broadband internal header format:

uint32_t n_az
uint32_t n_f
uint32_t precision_factor
int64_t  az[ 0 ] * precision_factor
  ...
int64_t  az[ n_az-1 ] * precision_factor
int64_t  f[ 0 ] * precision_factor
  ...
int64_t  f[ n_f-1 ] * precision_factor
[ body ]
*/
void NCPA::EPadeSolver::write_broadband_header( std::string filename, double *az_vec, size_t n_az, 
	double *f_vec, size_t n_f, unsigned int precision_factor ) {

	size_t i = 0;

	// open the file, truncating it if it exists
	std::ofstream ofs( filename, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary );
	if (!ofs.good()) {
		throw std::runtime_error( "Error opening file to initialize:" + filename );
	}

	size_t buf_size = n_az;
	if (n_f > buf_size) {
		buf_size = n_f;
	}
	int64_t *buffer = new int64_t[ buf_size ];
	std::memset( buffer, 0, buf_size * sizeof( int64_t ) );

	// write header starting with vector sizes and multiplicative factor
	uint32_t holder = n_az;
	ofs.write( (char*)(&holder), sizeof( uint32_t ) );
	holder = n_f;
	ofs.write( (char*)(&holder), sizeof( uint32_t ) );
	holder = precision_factor;
	ofs.write( (char*)(&holder), sizeof( uint32_t ) );

	for (i = 0; i < n_az; i++) {
		buffer[ i ] = (int64_t)std::lround( az_vec[ i ] * (double)precision_factor );
	}
	ofs.write( (char*)buffer, n_az * sizeof( int64_t ) );
	std::memset( buffer, 0, buf_size * sizeof( int64_t ) );
	for (i = 0; i < n_f; i++) {
		buffer[ i ] = (int64_t)std::lround( f_vec[ i ] * (double)precision_factor );
	}
	ofs.write( (char*)buffer, n_f * sizeof( int64_t ) );
	ofs.close();

	delete [] buffer;
}

/*
Broadband body format:
foreach (az)
  foreach (freq)
    int64_t  az                       * precision_factor
    int64_t  freq                     * precision_factor
    uint32_t n_z
    uint32_t n_range
    int64_t  z[ 0 ]                   * precision_factor
      ...
    int64_t  z[ n_z-1 ]               * precision_factor
    int64_t  range[ 0 ]               * precision_factor
      ...
    int64_t  range[ n_range-1 ]       * precision_factor
    int64_t  Re{ TL[ z[0] ][ r[0] ] } * precision_factor
	int64_t  Im{ TL[ z[0] ][ r[0] ] } * precision_factor
	int64_t  Re{ TL[ z[0] ][ r[1] ] } * precision_factor
	int64_t  Im{ TL[ z[0] ][ r[1] ] } * precision_factor
	  ...
	int64_t  Re{ TL[ z[0] ][ r[n_range-1] ] } * precision_factor
	int64_t  Im{ TL[ z[0] ][ r[n_range-1] ] } * precision_factor
	int64_t  Re{ TL[ z[1] ][ r[0] ] } * precision_factor
	int64_t  Im{ TL[ z[1] ][ r[0] ] } * precision_factor
	int64_t  Re{ TL[ z[1] ][ r[1] ] } * precision_factor
	int64_t  Im{ TL[ z[1] ][ r[1] ] } * precision_factor
	  ...
*/
void NCPA::EPadeSolver::write_broadband_results( std::string filename, double this_az, double this_f, 
	double *r_vec, size_t n_r, double *z_vec, size_t n_z, std::complex< double > **tloss_mat, 
	unsigned int precision_factor ) {

	n_r--;    // last range step is invalid

	std::ofstream ofs( filename, std::ofstream::out | std::ofstream::app | std::ofstream::binary );
	if (!ofs.good()) {
		throw std::runtime_error( "Error opening file to append: " + filename );
	}

	// write az, freq, n_z, n_range
	int64_t holder = (int64_t)std::lround( this_az * (double)precision_factor );
	ofs.write( (char*)(&holder), sizeof( int64_t ) );
	holder = (int64_t)std::lround( this_f * (double)precision_factor );
	ofs.write( (char*)(&holder), sizeof( int64_t ) );
	uint32_t uholder = (uint32_t)n_z;
	ofs.write( (char*)(&uholder), sizeof( uint32_t ) );
	uholder = (uint32_t)n_r;
	ofs.write( (char*)(&uholder), sizeof( uint32_t ) );

	// z and r sizes and vectors
	size_t buf_size = n_r;
	if (n_z > buf_size) {
		buf_size = n_z;
	}
	int64_t *buffer = new int64_t[ buf_size ];
	std::memset( buffer, 0, buf_size * sizeof( int64_t ) );
	size_t i, j;
	for (i = 0; i < n_z; i++) {
		buffer[ i ] = (int64_t)std::lround( z_vec[ i ] * (double)precision_factor );
	}
	ofs.write( (char*)buffer, n_z * sizeof( int64_t ) );
	for (i = 0; i < n_r; i++) {
		buffer[ i ] = (int64_t)std::lround( r_vec[ i ] * (double)precision_factor );
	}
	ofs.write( (char*)buffer, n_r * sizeof( int64_t ) );
	for (i = 0; i < n_z; i++) {
		for (j = 0; j < n_r; j++) {
			holder = (int64_t)std::lround( tloss_mat[ i ][ j ].real() * (double)precision_factor );
			ofs.write( (char *)(&holder), sizeof( int64_t ) );
			holder = (int64_t)std::lround( tloss_mat[ i ][ j ].imag() * (double)precision_factor );
			ofs.write( (char *)(&holder), sizeof( int64_t ) );
		}
	}
	ofs.close();
	delete [] buffer;
}