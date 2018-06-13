timestep <- "mon";
tFinal<- 120;
realizations = 100;

agingRate = 1 - 0.001


###############################
# Neighborhood specifications
###############################
numHoods=1


PacificPalisades = list(	mu = 0.001,
					sigma = 0.001,
					initialVal = 500000,
					hip_p=.05,
					hipVal_sigma=0.05,
					index = 1
					)
PacificPalisades$hipVal_mu	= (1-agingRate) / PacificPalisades$hip_p
						

#hoodNames = c(	"PacificPalisades"	)
neighborhoods=list(	PacificPalisades	)

####################
# Model Mechanics
####################

# Function to run one realization:
runReal = function()
	{

	###############################
	# Appreciation Processes
	###########################
	arrayDim = c(tFinal, numHoods)

	# The random process, Step-Wise by time increments
	appStep = array(
		c(	rnorm(tFinal*numHoods)),
		dim	=	arrayDim
		)
	for (j in 1:numHoods) {
		appStep[,j] =  appStep[,j] * neighborhoods[[j]]$sigma +
			 neighborhoods[[j]]$mu
		appStep[,j] = exp(appStep[,j])
		}

	# And the cumulative process
	appCumul = appStep
	for (i in 2:tFinal){
		appCumul[i,] = appCumul[i,] * appCumul[i-1,]
		}


	###############################
	# Home Values
	###############################

	# Constructor function for a "Home" object
	makeHome = function(
			hood,
			initialValMult = 1
			)
		{
		home = list(
			hood		= hood,
			initialVal	= hood$initialVal * initialValMult,
			hips		= ( runif(tFinal) < hood$hip_p )
			)

		# home value at t=1:
		home$vals[1]	= appStep[1,hood$index] * 
						home$initialVal *
						agingRate
		home$hipVals[1] 	= home$vals[1] * home$hips[1] *
						rnorm(1,
							mean = hood$hipVal_mu,
							sd = hood$hipVal_sigma
							)
		home$vals[1] 	= home$vals[1] + home$hipVals[1]
		
		# and at all t > 1:
		for (i in 2:tFinal)
			{
			home$vals[i]	= appStep[i,hood$index] * 
							home$vals[i-1] *
							agingRate
			if (home$hips[i] == TRUE)
				{
				home$hipVals[i]	= rnorm(1,
								mean = hood$hipVal_mu,
								sd = hood$hipVal_sigma) *
								home$vals[i]
				}
			else
				{
				home$hipVals[i]	= 0
				}
			home$vals[i]	= home$vals[i] + home$hipVals[i]
			}
		#return
		home
		}
	
	homes = list(
		home1 = makeHome(PacificPalisades)
		)
	for (hood in neighborhoods)
		{
		hood$appCumul = appCumul[,hood$index]
		}
	realization = list(
		homes			= homes,
		neighborhoods	= neighborhoods
		)
	

	#return
	realization
	}

r = list()
for (i in 1:realizations)
	{
	r[[i]] = runReal()
	if (i==1) plot(r[[i]]$homes$home1$vals)
	else points(r[[i]]$homes$home1$vals)
	}

	

