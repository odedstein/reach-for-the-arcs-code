if(TARGET PoissonRecon)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    PoissonRecon
    GIT_REPOSITORY https://github.com/mkazhdan/PoissonRecon.git
    GIT_TAG 6439de117e19f1d501d80a369e3b4069a2aa6644
)
FetchContent_MakeAvailable(PoissonRecon)