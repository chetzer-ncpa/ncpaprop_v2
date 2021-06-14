#include "parameterset.h"
#include "pebb_parameters.h"
#include <string>
#include <iostream>

void NCPA::configure_pebb_parameter_set( NCPA::ParameterSet *ps ) {

	NCPA::ParameterTest *test = NULL;

	// general configuration
	// set up expected commands
	ps->setStrict( true );
	ps->setComments( "#%" );
	ps->setDelimiters( "=: " );

	// Add header instructions
	ps->addHeaderTextVerbatim( "----------------------------------------------------------------------------" );
	ps->addHeaderTextVerbatim( "|                             NCPA Infrasound                              |" );  	
	ps->addHeaderTextVerbatim( "|                              PE Broadband                                |" );
	ps->addHeaderTextVerbatim( "|                      Based on: Pade PE - see ePape                       |" ); 
	ps->addHeaderTextVerbatim( "----------------------------------------------------------------------------" );	
 	ps->addBlankHeaderLine();
	
 	ps->addHeaderText( "To propagate a pulse, two steps must be completed:" );
 	ps->addHeaderText( "1. A set of transfer functions must be calculated using the option --broadband in ePape." );
 	ps->addHeaderText( "2. Pulse propagation is calculated for a selected source type: " );
 	ps->setHeaderIndent( 2 );
 	ps->addHeaderText( "--source impulse                           : Delta function" );
 	ps->addHeaderText( "--source pulse1                            : Built-in pulse type 1" );
 	ps->addHeaderText( "--source pulse2                            : Built-in pulse type 2" );
 	ps->addHeaderText( "--source spectrum --source_file <filename> : User-supplied spectrum file" );
 	ps->addHeaderText( "    Format: Freq   Re[ spec(f) ]  Im[ spec(f) ]" );
 	ps->addHeaderText( "--source waveform --source_file <filename> : User-supplied waveform file" );
 	ps->addHeaderText( "    Format: Time   Amplitude" );
 	ps->resetHeaderIndent();
	ps->addBlankHeaderLine();
	ps->addHeaderText("The options below can be specified in a colon-separated file \"pebb.param\" or at the command line. Command-line options override file options.");

	// Parameter descriptions
	ps->addParameter( new FlagParameter( "help" ) );
	ps->addParameter( new FlagParameter( "h" ) );
	ps->addParameterDescription( "Options Control", "--help", "Prints help test" );

	ps->addParameter( new StringParameter( "paramfile", "pebb.param") );
	ps->addParameterDescription( "Options Control", "--paramfile", "Parameter file name [pebb.param]" );

	ps->addParameter( new FlagParameter( "printparams" ) );
	ps->addParameterDescription( "Options Control", "--printparams", "Print parameter summary to screen" );

	// I/O
	ps->addParameter( new NCPA::StringParameter( "input_tloss_file", "" ) );
	ps->addTest( new NCPA::RequiredTest( "input_tloss_file" ) );
	ps->addParameterDescription( "Input/Output", "--input_tloss_file", "File to read transmission loss results from [required]" );

	ps->addParameter( new NCPA::StringParameter( "output_waveform_file", "" ) );
	ps->addTest( new NCPA::RequiredTest( "output_waveform_file" ) );
	ps->addParameterDescription( "Input/Output", "--output_waveform_file", "File to write resultant waveform to [required]" );

	ps->addParameter( new NCPA::StringParameter( "source", "impulse" ) );
	test = ps->addTest( new NCPA::StringSetTest( "source" ) );
	test->addStringParameter( "impulse" );
	test->addStringParameter( "pulse1" );
	test->addStringParameter( "pulse2" );
	test->addStringParameter( "waveform" );
	test->addStringParameter( "spectrum" );
	ps->addParameterDescription( "Calculation", "--source", "Source type.  Options include: {impulse,pulse1,pulse2,spectrum,waveform} [impulse]" );

	ps->setParameterIndent( 3 * DEFAULT_PARAMETER_INDENT );
	ps->addParameter( new NCPA::StringParameter( "source_file", "" ) );
	ps->addParameterDescription( "Calculation", "--source_file", "File containing the source spectrum or waveform, if applicable" );
	ps->addParameter( new NCPA::FloatParameter( "f_center", -1.0 ) );
	ps->addParameterDescription( "Calculation", "--f_center", "Center frequency for pulse1 or pulse2 options.  Must be <= f_max/5 [f_max/5]");
	ps->setParameterIndent( 2 * DEFAULT_PARAMETER_INDENT );

	ps->addParameter( new NCPA::IntegerParameter( "nfft", 0 ) );
	ps->addTest( new NCPA::IntegerGreaterThanOrEqualToTest( "nfft", 0 ) );
	ps->addParameterDescription( "Calculation", "--nfft", "Number of FFT points [4*f_max/f_step]");
	
	ps->addParameter( new NCPA::FloatParameter( "max_celerity", 340.0 ) );
	ps->addTest( new NCPA::IntegerGreaterThanTest( "max_celerity", 0 ) );
	ps->addParameterDescription( "Calculation", "--max_celerity", "Maximum celerity for calculation [340.0]");
	
	ps->addParameter( new NCPA::StringParameter( "receiver", "single" ) );
	test = ps->addTest( new NCPA::StringSetTest( "receiver" ) );
	test->addStringParameter( "single" );
	test->addStringParameter( "multiple" );
	ps->addParameterDescription( "Calculation", "--receiver", "Receiver type {single,multiple} [single]" );

	ps->setParameterIndent( 3 * DEFAULT_PARAMETER_INDENT );
	ps->addParameter( new NCPA::FloatParameter( "range_km", 0.0 ) );
	ps->addParameterDescription( "Calculation", "--range_km", "Propagation range in km for a single receiver");
	ps->addParameter( new NCPA::FloatParameter( "start_range_km", 0.0 ) );
	ps->addParameterDescription( "Calculation", "--start_range_km", "Starting propagation range in km for multiple receivers");
	ps->addParameter( new NCPA::FloatParameter( "end_range_km", 0.0 ) );
	ps->addParameterDescription( "Calculation", "--end_range_km", "Ending propagation range in km for multiple receivers");
	ps->addParameter( new NCPA::FloatParameter( "range_step_km", 0.0 ) );
	ps->addParameterDescription( "Calculation", "--range_step_km", "Propagation range step in km for multiple receivers");
	ps->resetParameterIndent();

	// Footer with file formats and sample commands
	ps->addBlankFooterLine();
	ps->addFooterText("OUTPUT Files:  Format description (column order):");
	ps->addFooterTextVerbatim("  <waveform file>          r[km]  t[s]  P" );
	ps->addBlankFooterLine();
	ps->addFooterText("Examples (run from 'samples' directory):");
	ps->setFooterIndent( 4 );
	ps->setFooterHangingIndent( 4 );
	ps->setCommandMode( true );
	ps->addFooterText("../bin/ePape --singleprop --starter self --atmosfile NCPA_canonical_profile_trimmed.dat --azimuth 90 --maxrange_km 500 --broadband --f_min 0.01 --f_step 0.01 --f_max 0.5" );
	ps->addBlankFooterLine();
	ps->addFooterTextVerbatim( " --then--" );
	ps->addFooterText("../bin/PEBB --input_tloss_file tloss_broadband.bin --output_waveform_file pebb_waveform.dat --receiver single --source impulse --range_km 240 --nfft 4096" );
	// ps->addBlankFooterLine();
	// ps->addFooterText("../bin/ModBB --propagation --input_dispersion_file myDispersionFile.dat --output_waveform_file mywavf.dat --receiver multiple --start_range_km 240 --end_range_km 300 --range_step_km 20 --source waveform --source_file source_waveform_input_example.dat" );
	ps->setFooterHangingIndent( 0 );
	ps->setCommandMode( false );
	ps->resetFooterIndent();
}
