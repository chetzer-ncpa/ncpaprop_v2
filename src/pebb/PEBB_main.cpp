#include <complex>
#include <string>
#include <ctime>
#include "pebb_parameters.h"
#include "parameterset.h"
#include "Atmosphere1D.h"
#include "BroadbandPropagator.h"
#include "PEBroadbandPropagator.h"

using namespace NCPA;
using namespace std;



int main( int argc, char **argv ) {

	// object to process the options
	ParameterSet *param = new ParameterSet();
	configure_pebb_parameter_set( param );
	param->parseCommandLine( argc, argv );

	// check for help text
	if (param->wasFound( "help" ) || param->wasFound("h") ) {
		param->printUsage( cout );
		return 1;
	}

	// See if an options file was specified
	string paramFile = param->getString( "paramfile" );
	param->parseFile( paramFile );

	// parse command line again, to override file options
	param->parseCommandLine( argc, argv );

	// see if we want a parameter summary
	if (param->wasFound( "printparams" ) ) {
		param->printParameters();
	}

	// run parameter checks
	if (! param->validate() ) {
		cout << "Parameter validation failed:" << endl;
		param->printFailedTests( cout );
		return 0;
	}

	// set the timer to measure running time
	std::time_t tm1 = std::time(NULL);

	BroadbandPropagator *prop = new PEBroadbandPropagator( param );
	prop->calculate_waveform();

	std::time_t tm2 = std::time(NULL);
	cout << "\nRun duration: " << difftime(tm2,tm1) << " seconds." << endl;
	cout << " ... main() broadband version is done." << endl;

	delete prop;
	delete param;

	return 0;


}