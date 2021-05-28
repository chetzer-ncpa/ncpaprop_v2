#ifndef NCPAPROP_EPADE_PE_H_INCLUDED
#define NCPAPROP_EPADE_PE_H_INCLUDED

#include <vector>

#include "petscksp.h"

#include "AtmosphericTransferFunctionSolver.h"
#include "Atmosphere1D.h"
#include "Atmosphere2D.h"
#include "parameterset.h"

#define NCPAPROP_EPADE_PE_FILENAME_1D "tloss_1d.pe"
#define NCPAPROP_EPADE_PE_FILENAME_2D "tloss_2d.pe"
#define NCPAPROP_EPADE_PE_FILENAME_MULTIPROP "tloss_multiprop.pe"
#define NCPAPROP_EPADE_PE_FILENAME_BROADBAND "tloss_broadband.bin"

namespace NCPA {

	class EPadeSolver : public AtmosphericTransferFunctionSolver {

	public:
		EPadeSolver( NCPA::ParameterSet *param );
		~EPadeSolver();
		int solve();
		void output1DTL( std::string filename, bool append = false );
		void output2DTL( std::string filename );

	protected:

		// solve using the appropriate method
		int solve_with_topography();
		int solve_without_topography();

		// functions to perform the various intermediate calculations
		// int epade( int order, double k0, double dr, std::vector<PetscScalar> *P, std::vector<PetscScalar> *Q,
		// 	bool starter = false );
		int calculate_pade_coefficients( std::vector<PetscScalar> *c, 
			int n_numerator, int n_denominator, std::vector<PetscScalar> *numerator_coefficients,
			std::vector<PetscScalar> *denominator_coefficients );
		int make_q_powers( NCPA::Atmosphere2D *atm, int NZvec, double *zvec, double r, 
			std::complex<double> *k, 
			double k0, double h2, double ground_height, std::complex<double> ground_impedence, 
			std::complex<double> *n, size_t nqp, int boundary_index, const Mat &last_q, Mat *qpowers );
		int generate_polymatrices( Mat *qpowers, int npade, int NZ, 
			std::vector< std::complex< double > > &P, std::vector< std::complex< double > > &Q,
			Mat *B, Mat *C );
		int generate_polymatrix( Mat *qpowers, int Qpowers_size, int NZ, 
			std::vector< std::complex< double > > &T, Mat *B );
		int create_polymatrix_vector( size_t nterms, const Mat *Q, Mat **qpowers );
		int delete_polymatrix_vector( size_t nterms, Mat **qpowers );
		int build_operator_matrix_with_topography( NCPA::Atmosphere2D *atm, int NZvec, double *zvec, 
			double r, std::complex<double> *k, double k0, double h2, double z_s,
			std::complex<double> impedence_factor, std::complex<double> *n, 
			int boundary_index, const Mat &last_q, Mat *q );
		int build_operator_matrix_without_topography( 
			int NZvec, double *zvec, double k0, double h2, 
			std::complex<double> impedence_factor, std::complex<double> *n, size_t nqp, 
			int boundary_index, Mat *q );

		// functions for recurrence relations of various Taylor series
		std::vector<PetscScalar> taylor_exp_id_sqrt_1pQ_m1( int N, double delta );
		std::vector<PetscScalar> taylor_1pQ_n025( int N );
		std::vector<PetscScalar> taylor_1pQ_025( int N );
		std::vector<PetscScalar> taylor_1pQpid_n025( int N, double delta );

		// approximation functions
		int approximate_sqrt_1pQ( int NZvec, const Mat *Q, PetscInt Ji, Vec *vecBelow, Vec *vecAbove, PetscInt *nonzeros );

		// functions to calculate the various starter fields
		int get_starter_gaussian( size_t NZ, double *z, double zs, double k0, int ground_index, Vec *psi );
		// int get_starter_self( size_t NZ, double *z, double z_source, double z_ground, double k0, 
		// 	Mat *qpowers, size_t npade, Vec *psi );
		int get_starter_self_revised( size_t NZ, double *z, double z_source, double rr, 
			double z_ground, double k0, Mat *qpowers, size_t npade, Vec *psi );

		// functions to calculate atmospheric parameters
		void absorption_layer( double lambda, double *z, int NZ, double *layer );
		void fill_atm_vector_relative( NCPA::Atmosphere2D *atm, double range, int NZvec, double *zvec, 
			std::string key, double groundheight, double *vec );
		void fill_atm_vector_absolute( NCPA::Atmosphere2D *atm, double range, int NZvec, double *zvec, 
			std::string key, double fill_value, double *vec );
		void calculate_atmosphere_parameters( NCPA::Atmosphere2D *atm, int NZvec, double *z_vec, 
			double r, double z_g, bool use_lossless, bool use_top_layer, double freq, bool use_absolute_z,
			double &k0, double &c0, double *c_vec, double *a_vec, std::complex<double> *k_vec, 
			std::complex<double> *n_vec );

		void set_1d_output( bool tf );
		void write_broadband_header( std::string filename, double *az_vec, size_t n_az, 
			double *f_vec, size_t n_f, unsigned int precision_factor );
		void write_broadband_results( std::string filename, double this_az, double this_f, 
			double *r_vec, size_t n_r, double *z_vec, size_t n_z, std::complex< double > **tloss_mat, 
			unsigned int precision_factor );

		double *z = NULL, *z_abs = NULL, *r = NULL, *f = NULL, calc_az;
		std::complex< double > **tl;
		int *zgi_r = NULL;   // ground height index
		double freq;         // current active frequency
		double *azi;
		int NZ, NR, NR_requested, NAz, NF;
		double dz;
		int npade;
		bool use_atm_1d = false, use_atm_2d = false, use_atm_toy = false, use_topo = false;
		bool z_ground_specified = false, lossless = false, top_layer = true;
		bool multiprop = false, write1d = true, write2d = false, calculate_attn = true;
		bool broadband = false;
		double r_max;    // range limits
		double z_max, z_min, z_ground, z_bottom;  // atmosphere profile limits
		double zs, zr;  // source height, receiver height
		double c_underground;
		std::complex<double> user_ground_impedence;
		bool user_ground_impedence_found = false;
		//double zrcv;
		//std::string gnd_imp_model;
		std::string starter;
		std::string attnfile;

		std::vector< double > zt;
		std::vector< int > zti;
		int nzplot;

		double absorption_layer_mu = 0.01;

		//NCPA::Atmosphere1D *atm_profile;
		NCPA::Atmosphere2D *atm_profile_2d;


	};

}







#endif