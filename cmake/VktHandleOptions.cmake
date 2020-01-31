# This file is distributed under the MIT license.
# See the LICENSE file for details.


#---------------------------------------------------------------------------------------------------
# Module to handle VKT_* options
#

if(MSVC)

elseif(__COMPILER_PGI)

else() # GNU, Clang, etc.

    #------------------------------------------------------
    # Enable warnings
    #
    
    add_definitions(-Wmissing-braces)
    add_definitions(-Wsign-compare)
    add_definitions(-Wwrite-strings)
    add_definitions(-Woverloaded-virtual)
    add_definitions(-Wundef)

    if(VKT_ENABLE_WARNINGS)
        add_definitions(-Wall -Wextra)

        if(VKT_ENABLE_PEDANTIC)
            add_definitions(-pedantic)
        endif()
    endif()


    #------------------------------------------------------
    # Selectively disable warnings
    #

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0)
            # Disable warnings like:
            # warning: ignoring attributes on template argument __m128i
            add_definitions(-Wno-ignored-attributes)
        endif()
    endif()
endif()
