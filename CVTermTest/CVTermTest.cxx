/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#define BOOST_EXCEPTION_DISABLE 1
#include "itkRegionBasedLevelSetFunction.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkLevelSetDomainMapImageFilter.h"
#include "itkLevelSetEquationCurvatureTerm.h"
#include "itkLevelSetEquationContainer.h"
#include "itkAtanRegularizedHeavisideStepFunction.h"
#include "itkLevelSetEvolution.h"
#include "itkLevelSetEquationTermContainer.h"
#include "itkWhitakerSparseLevelSetImage.h"
#include "itkLevelSetEvolutionNumberOfIterationsStoppingCriterion.h"
#include "itkBinaryImageToLevelSetImageAdaptor.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkAtanRegularizedHeavisideStepFunction.h"
#include "itkBinaryImageToLevelSetImageAdaptor.h"
#include "itkLevelSetEvolutionNumberOfIterationsStoppingCriterion.h"
#include "itkLevelSetEquationChanAndVeseInternalTerm.h"
#include "itkLevelSetContainer.h"
#include "itkLevelSetEquationChanAndVeseExternalTerm.h"
#include <time.h>
#include <math.h>


int main( int argc, char* argv[] )
{
 
  // Print help message
  if (argc < 8)
    {
    std::cout << "========================================================================="<< std::endl;
	std::cout << "Perform piecewise constant regional levelset segmentation, i.e chan vese levelset method" << std::endl ;
    std::cout << "Algorithm by Chan and Vese"<<std::endl;
    std::cout << "Author: Hui Tang"<<std::endl;
    std::cout << "========================================================================="<< std::endl;
    std::cout << "Example:CVTermTest.exe  initial.mhd originalImage.mhd output.mhd internalWeight externalWeight curvatureWeight iterationTime RSM"<< std::endl;
	return EXIT_FAILURE;
    }

  // String for in- and ouput file
  std::string initialImageN;
  std::string originalImageN;
  std::string outputImageN;


  // Get intensity file
  if( argv[1] )
    {
    initialImageN = argv[1];
    std::cout << "inputImage: " << initialImageN<< std::endl;
    }
  else
    {
    std::cout << "Error: no initial image provided" << std::endl;
    return EXIT_FAILURE;
    }

  const unsigned int Dimension = 3;
  typedef float                                    InputPixelType;
  typedef itk::Image< InputPixelType, Dimension >  InputImageType;

  typedef itk::ImageFileReader< InputImageType >            ReaderType;
  ReaderType::Pointer initialReader = ReaderType::New();
  initialReader->SetFileName( initialImageN );
  initialReader->Update();
  InputImageType::Pointer initial = initialReader->GetOutput();
  std::vector<float> imgExt(3, 0);
  imgExt[0] = initial->GetBufferedRegion().GetSize()[0];
  imgExt[1] = initial->GetBufferedRegion().GetSize()[1];
  imgExt[2] = initial->GetBufferedRegion().GetSize()[2];
  std::cout << "initial Image Extent " << imgExt[0] << ", " << imgExt[1] << ", " << imgExt[2] << std::endl;
  // Get intensity file

  // Get intensity file
  if (argv[2])
    {
    originalImageN = argv[2];
    std::cout << "originalImage: " << originalImageN<< std::endl;
    }
  else
    {
    std::cout << "Error: no original image provided" << std::endl;
    return EXIT_FAILURE;
    }

  ReaderType::Pointer originalReader = ReaderType::New();
  originalReader->SetFileName( originalImageN);
  originalReader->Update();

  InputImageType::Pointer original = originalReader->GetOutput();
  std::vector<float> imgExt2(3, 0);
  imgExt2[0] = original->GetBufferedRegion().GetSize()[0];
  imgExt2[1] = original->GetBufferedRegion().GetSize()[1];
  imgExt2[2] = original->GetBufferedRegion().GetSize()[2];
  std::cout << "original image Extent " << imgExt2[0] << ", " << imgExt2[1] << ", " << imgExt2[2] << std::endl;
  if (imgExt[0]!=imgExt2[0]||imgExt[1]!=imgExt2[1]||imgExt[2]!=imgExt2[2])
    {
    std::cout << "image size should be the same!" << std::endl;
    return EXIT_FAILURE;
    }

  if (argv[3])
    {
    outputImageN = argv[3];
    std::cout << "outputImage: " << outputImageN<< std::endl;
    }
  else
    {
    std::cout << "Error: no output image name provided" << std::endl;
    return EXIT_FAILURE;
    }

  typedef itk::WhitakerSparseLevelSetImage < InputPixelType, Dimension > SparseLevelSetType;
  typedef itk::BinaryImageToLevelSetImageAdaptor< InputImageType,
	  SparseLevelSetType> BinaryToSparseAdaptorType;

  BinaryToSparseAdaptorType::Pointer adaptor = BinaryToSparseAdaptorType::New();
  adaptor->SetInputImage( initial );
  adaptor->Initialize();

  typedef  SparseLevelSetType::Pointer SparseLevelSetTypePointer;
  SparseLevelSetTypePointer levelset = adaptor->GetLevelSet();

  typedef itk::IdentifierType         IdentifierType;
  typedef std::list< IdentifierType > IdListType;

  IdListType listIds;
  listIds.push_back( 1 );

  typedef itk::Image< IdListType, Dimension >               IdListImageType;
   IdListImageType::Pointer idimage = IdListImageType::New();
  idimage->SetRegions( initial->GetLargestPossibleRegion() );
  idimage->Allocate();
  idimage->FillBuffer( listIds );

  typedef itk::Image< short, Dimension >                     CacheImageType;
  typedef itk::LevelSetDomainMapImageFilter< IdListImageType, CacheImageType >
	  DomainMapImageFilterType;
   DomainMapImageFilterType::Pointer domainMapFilter = DomainMapImageFilterType::New();
  domainMapFilter->SetInput( idimage );
  domainMapFilter->Update();

  // Define the Heaviside function
  typedef  SparseLevelSetType::OutputRealType LevelSetOutputRealType;

  typedef itk::AtanRegularizedHeavisideStepFunction< LevelSetOutputRealType,LevelSetOutputRealType > HeavisideFunctionType;
   HeavisideFunctionType::Pointer heaviside = HeavisideFunctionType::New();
  heaviside->SetEpsilon( 1.5 );

  // Insert the levelsets in a levelset container
  typedef itk::LevelSetContainer< IdentifierType, SparseLevelSetType >
	  LevelSetContainerType;
  typedef itk::LevelSetEquationTermContainer< InputImageType, LevelSetContainerType >
	  TermContainerType;
   LevelSetContainerType::Pointer lsContainer = LevelSetContainerType::New();
  lsContainer->SetHeaviside( heaviside );
  lsContainer->SetDomainMapFilter( domainMapFilter );

  lsContainer->AddLevelSet( 0, levelset );

  std::cout << std::endl;
  std::cout << "Level set container created" << std::endl;


  typedef itk::LevelSetEquationChanAndVeseInternalTerm< InputImageType,
	  LevelSetContainerType > CVTermTypeIn;

  CVTermTypeIn::Pointer cvTermIn0 = CVTermTypeIn::New();
  cvTermIn0->SetInput( original  );
  cvTermIn0->SetCoefficient( atof(argv[4])  );
  std::cout<<"InternalCoefficient:"<<atof(argv[4])<<std::endl;
  cvTermIn0->SetCurrentLevelSetId( 0 );
  cvTermIn0->SetLevelSetContainer( lsContainer );

  typedef itk::LevelSetEquationChanAndVeseExternalTerm< InputImageType,
	  LevelSetContainerType > CVTermTypeEx;
  CVTermTypeEx::Pointer cvTermEx0 = CVTermTypeEx::New();
  cvTermEx0->SetCoefficient(  atof(argv[5]) );
  std::cout<<"ExternalCoefficient:"<<atof(argv[5]) <<std::endl;
  cvTermEx0->SetCurrentLevelSetId( 0 );
  cvTermEx0->SetLevelSetContainer( lsContainer );

  typedef itk::LevelSetEquationCurvatureTerm< InputImageType,
      LevelSetContainerType > CurvatureTermType;

  CurvatureTermType::Pointer curvatureTerm0 =  CurvatureTermType::New();
  curvatureTerm0->SetCoefficient( atof(argv[6])  );
  std::cout<<"CurvatureWeight:"<<atof(argv[6])<<std::endl;
  curvatureTerm0->SetCurrentLevelSetId( 0 );
  curvatureTerm0->SetLevelSetContainer( lsContainer );





  // put the curvature term here!

  // **************** CREATE ALL EQUATIONS ****************

  // Create Term Container which corresponds to the combination of terms in the PDE.

  TermContainerType::Pointer termContainer0 = TermContainerType::New();

  termContainer0->SetInput( original  );
  termContainer0->SetCurrentLevelSetId( 0 );
  termContainer0->SetLevelSetContainer( lsContainer );
  termContainer0->AddTerm( 0, cvTermIn0  );
  termContainer0->AddTerm( 1, cvTermEx0  );
  termContainer0->AddTerm( 2, curvatureTerm0 );


  typedef itk::LevelSetEquationContainer< TermContainerType > EquationContainerType;
  EquationContainerType::Pointer equationContainer = EquationContainerType::New();
  equationContainer->AddEquation( 0, termContainer0 );
  equationContainer->SetLevelSetContainer( lsContainer );

  typedef itk::LevelSetEvolutionNumberOfIterationsStoppingCriterion< LevelSetContainerType >
    StoppingCriterionType;
  StoppingCriterionType::Pointer criterion = StoppingCriterionType::New();
  criterion->SetNumberOfIterations(    atof(argv[7]) );
  std::cout<<"NumberOfIterations:"<<atof(argv[7])<<std::endl;

  criterion->SetRMSChangeAccumulator( atof(argv[8]));
  std::cout<<"RMSChangeAccumulator:"<<atof(argv[8])<<std::endl;


   typedef itk::LevelSetEvolution< EquationContainerType, SparseLevelSetType > LevelSetEvolutionType;
   LevelSetEvolutionType::Pointer evolution = LevelSetEvolutionType::New();

  evolution->SetEquationContainer( equationContainer );
  evolution->SetStoppingCriterion( criterion );
  evolution->SetLevelSetContainer( lsContainer );

  try
    {
    evolution->Update();
    }
    catch ( itk::ExceptionObject& err )
    {
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    std::cout<<"ERROR::can not update evolution!!"<<std::endl;
    }

  InputImageType::Pointer outputImage = InputImageType::New();
  outputImage->SetRegions( original->GetLargestPossibleRegion() );
  outputImage->CopyInformation( original );
  outputImage->Allocate();
  outputImage->FillBuffer( 0 );

  typedef itk::ImageRegionIteratorWithIndex< InputImageType > OutputIteratorType;
  OutputIteratorType oIt( outputImage, outputImage->GetLargestPossibleRegion() );
  oIt.GoToBegin();

  InputImageType::IndexType idx;

  while( !oIt.IsAtEnd() )
    {
    idx = oIt.GetIndex();
    //oIt.Set( level_set->GetLabelMap()->GetPixel(idx) );
	oIt.Set( levelset->Evaluate(idx) );
    ++oIt;
    }

  typedef itk::ImageFileWriter< InputImageType >     OutputWriterType;
  OutputWriterType::Pointer writer = OutputWriterType::New();
  writer->SetFileName(outputImageN);
  //writer->SetInput( binary);
  writer->SetInput( outputImage );

  try
    {
    writer->Update();

    std::cout << "outputfile is saved as " << outputImageN<<std::endl;
    }
  catch ( itk::ExceptionObject& err )
    {
    std::cout << err << std::endl;
    }
  time_t rawtime1;
  struct tm * timeinfo1;
  time ( &rawtime1 );
  timeinfo1 = localtime ( &rawtime1 );
}
