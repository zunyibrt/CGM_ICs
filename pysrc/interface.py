######### interface base classes for potential and circular velocity
class Cooling:
    """interface for cooling function"""
    
    def LAMBDA(self,T,nH,Z,r):
        """cooling function"""
        assert(False)

class Metallicity_profile:
    def Z(self, r):
        assert(False)

class Boundary_Conditions:
    def outer_radius(self):
        assert(False)
    def outer_density(self):
        assert(False)
    def outer_temperature(self):
        assert(False)
        
            
class Potential: 
    """interface for gravitational potential"""
    def vc(self,r): 
        """circular velocity"""
        assert(False)
    def Phi(self,r):
        """gravitational potential"""
        assert(False)


        
    
