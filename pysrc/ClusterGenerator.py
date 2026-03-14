"""
Star Cluster Generator Module

Generates synthetic star clusters with power-law mass distribution 
and uniform spatial distribution for astrophysical simulations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as PathEffects
import cmasher as cmr

@dataclass
class ClusterParameters:
    """Configuration parameters for star cluster generation."""
    n_clusters: int = 1000
    mass_range: Tuple[float, float] = (1e4, 1e6)  # Solar masses
    alpha: float = 2.0  # Power-law index
    radius_kpc: float = 1.0  # Kiloparsecs
    
    @property
    def radius_pc(self) -> float:
        """Convert radius from kpc to parsecs."""
        return self.radius_kpc * 1000
    
    @property
    def mass_min(self) -> float:
        """Minimum mass in solar masses."""
        return self.mass_range[0]
    
    @property
    def mass_max(self) -> float:
        """Maximum mass in solar masses."""
        return self.mass_range[1]
    
    def validate(self) -> None:
        """Validate parameter values."""
        if self.n_clusters <= 0:
            raise ValueError("Number of clusters must be positive")
        if self.mass_min <= 0 or self.mass_max <= 0:
            raise ValueError("Mass values must be positive")
        if self.mass_min >= self.mass_max:
            raise ValueError("Minimum mass must be less than maximum mass")
        if self.alpha <= 0:
            raise ValueError("Power-law index must be positive")
        if self.radius_kpc <= 0:
            raise ValueError("Radius must be positive")


class MassDistribution:
    """Handles mass distribution calculations for star clusters."""
    
    @staticmethod
    def generate_power_law(n: int, m_min: float, m_max: float, 
                          alpha: float, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate masses following a power-law distribution n ∝ m^(-α).
        
        Uses inverse transform sampling for efficient generation.
        
        Parameters:
        -----------
        n : int
            Number of masses to generate
        m_min : float
            Minimum mass
        m_max : float
            Maximum mass
        alpha : float
            Power-law index
        seed : Optional[int]
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Array of generated masses
        """
        if seed is not None:
            np.random.seed(seed)
            
        u = np.random.random(n)
        alpha_minus_1 = 1 - alpha
        
        if abs(alpha_minus_1) < 1e-10:  # Special case for α ≈ 1
            masses = m_min * (m_max / m_min) ** u
        else:
            # General case: inverse transform for power law
            term1 = m_max ** alpha_minus_1
            term2 = m_min ** alpha_minus_1
            masses = (u * (term1 - term2) + term2) ** (1 / alpha_minus_1)
        
        return masses
    
    @staticmethod
    def calculate_expected_histogram(bin_edges: np.ndarray, n_total: int, 
                                    alpha: float) -> np.ndarray:
        """
        Calculate expected histogram counts for power-law distribution.
        
        Parameters:
        -----------
        bin_edges : np.ndarray
            Edges of histogram bins (in log space)
        n_total : int
            Total number of samples
        alpha : float
            Power-law index
            
        Returns:
        --------
        np.ndarray
            Expected counts per bin
        """
        expected = []
        
        for i in range(len(bin_edges) - 1):
            m_low = 10 ** bin_edges[i]
            m_high = 10 ** bin_edges[i + 1]
            
            if abs(alpha - 1) < 1e-10:
                integral = np.log(m_high / m_low)
            else:
                integral = (m_high ** (1 - alpha) - m_low ** (1 - alpha)) / (1 - alpha)
            
            expected.append(integral)
        
        expected = np.array(expected)
        return expected * n_total / expected.sum()


class SpatialDistribution:
    """Handles spatial distribution of star clusters."""
    
    @staticmethod
    def generate_uniform_disk(n: int, radius: float, 
                            seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate uniform random positions within a disk.
        
        Uses sqrt(r) transformation to avoid central clustering.
        
        Parameters:
        -----------
        n : int
            Number of positions to generate
        radius : float
            Disk radius
        seed : Optional[int]
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            x coordinates, y coordinates, radial distances
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Avoid center clustering with sqrt transformation
        r = np.sqrt(np.random.random(n)) * radius
        theta = np.random.random(n) * 2 * np.pi
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return x, y, r


class ClusterVisualizer:
    """Handles visualization of star cluster data."""
    
    def __init__(self, style: str = 'dark'):
        """
        Initialize visualizer with specified style.
        
        Parameters:
        -----------
        style : str
            Visualization style ('dark' or 'default')
        """
        self.style = style
        self.colors = {
            'dark': {
                'bg': '#0a0a1a',
                'text': '#90caf9',
                'title': '#64b5f6',
                'grid': '#64b5f6',
                'card_bg': '#2a2a3e',
                'card_edge': '#404060'
            },
            'default': {
                'bg': 'white',
                'text': 'black',
                'title': 'black',
                'grid': 'gray',
                'card_bg': '#f0f0f0',
                'card_edge': 'gray'
            }
        }
    
    def plot_distributions(self, clusters_df: pd.DataFrame, params: ClusterParameters,
                          stats: Dict[str, float], figsize: Tuple[int, int] = (16, 6),
                          save_fig: bool = False, filename: Optional[str] = None) -> None:
        """
        Create comprehensive visualization of cluster distributions.
        
        Parameters:
        -----------
        clusters_df : pd.DataFrame
            DataFrame containing cluster data
        params : ClusterParameters
            Generation parameters
        stats : Dict[str, float]
            Cluster statistics
        figsize : Tuple[int, int]
            Figure size
        save_fig : bool
            Whether to save the figure
        filename : Optional[str]
            Output filename if saving
        """
        # Set style
        if self.style == 'dark':
            plt.style.use('dark_background')
        
        colors = self.colors[self.style]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor(colors['bg'])
        
        # Plot spatial distribution
        self._plot_spatial_distribution(ax1, clusters_df, params, colors)
        
        # Plot mass distribution
        self._plot_mass_distribution(ax2, clusters_df, params, stats, colors)
        
        plt.tight_layout()
        
        if save_fig:
            if filename is None:
                filename = f'star_clusters_{params.n_clusters}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor=colors['bg'])
        
        plt.show()
        
        # Reset style
        plt.style.use('default')
    
    def _plot_spatial_distribution(self, ax, clusters_df: pd.DataFrame, 
                                  params: ClusterParameters, colors: Dict) -> None:
        """Plot spatial distribution with concentric circles."""
        ax.set_facecolor(colors['bg'])
        
        # Draw concentric circles
        circle_radii = np.linspace(200, params.radius_pc, 5)
        for radius in circle_radii:
            circle = Circle((0, 0), radius, fill=False, color=colors['grid'],
                          alpha=0.2, linewidth=1)
            ax.add_patch(circle)
        
        # Calculate sizes based on mass
        masses = clusters_df['mass_msun'].values
        sizes = np.maximum(10, np.log10(masses / 1e4) * 15 + 20)
        
        # Plot clusters
        scatter = ax.scatter(clusters_df['x_pc'], clusters_df['y_pc'],
                           c=np.log10(masses), cmap='cmr.bubblegum',
                           s=sizes, alpha=0.8, edgecolors='none')
        
        # Styling
        ax.set_title('Spatial Distribution', color=colors['title'], 
                    fontsize=14, fontweight='bold', pad=5)
        ax.set_aspect('equal')
        
        max_coord = params.radius_pc * 1.1
        ax.set_xlim(-max_coord, max_coord)
        ax.set_ylim(-max_coord, max_coord)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('log₁₀(Mass/M☉)', color=colors['text'], fontsize=11)
        cbar.outline.set_edgecolor(colors['title'])
        cbar.ax.tick_params(colors=colors['text'], labelsize=9)
        
        # Scale indicator
        ax.text(0, -max_coord, f'{params.radius_kpc} kpc radius',
               ha='center', va='top', color=colors['text'], 
               fontsize=11, style='italic')
    
    def _plot_mass_distribution(self, ax, clusters_df: pd.DataFrame,
                               params: ClusterParameters, stats: Dict[str, float],
                               colors: Dict) -> None:
        """Plot mass distribution histogram with statistics."""
        ax.set_facecolor(colors['bg'])
        
        log_masses = np.log10(clusters_df['mass_msun'])
        
        # Create histogram
        n_bins = 20
        bin_edges = np.linspace(np.log10(params.mass_min), 
                              np.log10(params.mass_max), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        counts, _, patches = ax.hist(log_masses, bins=bin_edges, alpha=0.8,
                                    edgecolor='#1a1a2e', linewidth=0.5)
        
        # Color gradient for bars
        for i, patch in enumerate(patches):
            color_ratio = i / (len(patches) - 1)
            color = plt.cm.get_cmap('cmr.bubblegum')(color_ratio)
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        # Add count labels
        for i, count in enumerate(counts):
            if count > 0:
                ax.text(bin_centers[i], count + max(counts) * 0.01, 
                       f'{int(count)}', ha='center', va='bottom',
                       color=colors['text'], fontsize=9,
                       path_effects=[PathEffects.withStroke(linewidth=1, 
                                                           foreground='black')])
        
        # Plot expected distribution
        expected = MassDistribution.calculate_expected_histogram(
            bin_edges, params.n_clusters, params.alpha
        )
        ax.plot(bin_centers, expected, '#228b22', linewidth=3,
               linestyle=':', label=f'n ∝ m$^{{-{params.alpha:.1f}}}$', alpha=0.9)
        
        # Add statistics cards
        self._add_stat_cards(ax, stats, colors)
        
        # Styling
        ax.set_xlabel('log₁₀(Mass/M☉)', color=colors['text'], fontsize=12)
        ax.set_ylabel('Number of Clusters', color=colors['text'], fontsize=12)
        ax.set_title('Mass Distribution', color=colors['title'], 
                    fontsize=14, fontweight='bold', pad=5)
        
        # Clean up axes
        ax.tick_params(colors=colors['text'], labelsize=10, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.set_yticks([])
        ax.set_xticks([4, 4.5, 5, 5.5, 6])
        ax.set_xlim(3.95, 6.05)
        
        # Legend
        legend = ax.legend(loc='upper right', frameon=False)
        for text in legend.get_texts():
            text.set_color(colors['text'])
    
    def _add_stat_cards(self, ax, stats: Dict[str, float], colors: Dict) -> None:
        """Add statistics cards to the plot."""
        # Total clusters card
        box1 = FancyBboxPatch((0.75, 0.55), 0.22, 0.13, 
                             transform=ax.transAxes,
                             boxstyle="round,pad=0.01", 
                             facecolor=colors['card_bg'],
                             alpha=0.8, edgecolor=colors['card_edge'], 
                             linewidth=1)
        ax.add_patch(box1)
        
        ax.text(0.86, 0.635, f"{stats['n_clusters']}", 
               transform=ax.transAxes,
               ha='center', va='center', color=colors['title'], 
               fontsize=18, fontweight='bold')
        ax.text(0.86, 0.585, "Total Clusters", transform=ax.transAxes,
               ha='center', va='center', color='#b0b0b0', fontsize=9)
        
        # Total mass card
        box2 = FancyBboxPatch((0.75, 0.35), 0.22, 0.13, 
                             transform=ax.transAxes,
                             boxstyle="round,pad=0.01", 
                             facecolor=colors['card_bg'],
                             alpha=0.8, edgecolor=colors['card_edge'], 
                             linewidth=1)
        ax.add_patch(box2)
        
        total_mass_formatted = f"{stats['total_mass_msun']/1e6:.1f}M"
        ax.text(0.86, 0.435, total_mass_formatted, transform=ax.transAxes,
               ha='center', va='center', color=colors['title'], 
               fontsize=18, fontweight='bold')
        ax.text(0.86, 0.385, "Total Mass (M☉)", transform=ax.transAxes,
               ha='center', va='center', color='#b0b0b0', fontsize=9)


class StarClusterGenerator:
    """
    Main class for generating synthetic star clusters.
    
    This generator creates star clusters with:
    - Power-law mass distribution: n(m) ∝ m^(-α)
    - Uniform spatial distribution within a disk
    
    Example:
    --------
    >>> params = ClusterParameters(n_clusters=1000, alpha=2.0)
    >>> generator = StarClusterGenerator(params)
    >>> clusters = generator.generate_clusters(seed=42)
    >>> generator.plot_distributions()
    """
    
    def __init__(self, params: Optional[ClusterParameters] = None):
        """
        Initialize the star cluster generator.
        
        Parameters:
        -----------
        params : Optional[ClusterParameters]
            Configuration parameters. If None, uses defaults.
        """
        self.params = params or ClusterParameters()
        self.params.validate()
        self.clusters: Optional[pd.DataFrame] = None
        self.visualizer = ClusterVisualizer(style='dark')
    
    def _calculate_velocities(self, x: np.ndarray, y: np.ndarray, 
                            r: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate velocity components for counterclockwise circular motion with vertical kick.
        
        Parameters:
        -----------
        x : np.ndarray
            X coordinates in parsecs
        y : np.ndarray
            Y coordinates in parsecs
        r : np.ndarray
            Radial distances in parsecs
        n : int
            Number of clusters
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            vx, vy, vz velocity components and velocity magnitude
        """
        # Calculate velocity magnitude at fixed height (10 kpc)
        # Assumes combined.vc and cu_velocity are defined elsewhere in the code
        v_mag = 1.0 #(combined.vc(10*un.kpc)/cu_velocity).to('')
        
        # For counterclockwise motion: v_tangential = (-y, x) / r * v_mag
        # Avoid division by zero for clusters at the center
        vx = np.zeros_like(x)
        vy = np.zeros_like(y)
        
        non_zero = r > 0
        vx[non_zero] = -y[non_zero] / r[non_zero] * v_mag
        vy[non_zero] = x[non_zero] / r[non_zero] * v_mag
        
        # Add vertical velocity kick - normally distributed with mean 0, std 1
        vz = np.random.normal(0, 1, n)
        
        return vx, vy, vz, v_mag
    
    def generate_clusters(self, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate a complete set of star clusters.
        
        Parameters:
        -----------
        seed : Optional[int]
            Random seed for reproducibility
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing cluster properties:
            - id: Cluster identifier
            - mass_msun: Mass in solar masses
            - x_pc, y_pc: Cartesian coordinates in parsecs
            - r_pc: Radial distance from center in parsecs
            - vx, vy, vz: Velocity components (vx, vy for circular motion, vz is vertical kick)
            - v_mag: Velocity magnitude (in x-y plane)
        """
        # Set random seed for complete reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Generate masses
        masses = MassDistribution.generate_power_law(
            self.params.n_clusters,
            self.params.mass_min,
            self.params.mass_max,
            self.params.alpha
        )
        
        # Generate positions in x-y plane
        x, y, r = SpatialDistribution.generate_uniform_disk(
            self.params.n_clusters,
            self.params.radius_pc
        )
        
        # Generate z-coordinates: uniform between -10 and 10
        z = np.random.uniform(-10, 10, self.params.n_clusters)

        # Generate t_min values: uniform between 0 and 0.1
        t_min = np.random.uniform(0, 0.1, self.params.n_clusters)
        
        # Calculate velocities for counterclockwise motion with vertical kick
        vx, vy, vz, v_mag = self._calculate_velocities(x, y, r, self.params.n_clusters)
        
        # Create DataFrame
        self.clusters = pd.DataFrame({
            'id': range(1, self.params.n_clusters + 1),
            'mass_msun': masses,
            'x_pc': x,
            'y_pc': y,
            'z_pc': z,
            'r_pc': r,
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'v_mag': v_mag,
            't_min': t_min
        })
        
        return self.clusters
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Calculate summary statistics for the generated clusters.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary containing:
            - n_clusters: Total number of clusters
            - total_mass_msun: Total mass in solar masses
            - mean_mass_msun: Mean cluster mass
            - median_mass_msun: Median cluster mass
            - std_mass_msun: Standard deviation of masses
            - min_mass_msun: Minimum cluster mass
            - max_mass_msun: Maximum cluster mass
            - mean_radius_pc: Mean radial distance
            - max_radius_pc: Maximum radial distance
            - mass_range_decades: Mass range in decades (log scale)
            - mean_v_mag: Mean velocity magnitude (x-y plane)
            - std_v_mag: Standard deviation of velocity magnitude
            - mean_vz: Mean vertical velocity (should be ~0)
            - std_vz: Standard deviation of vertical velocity (should be ~1)
            - min_z_pc: Minimum z-coordinate
            - max_z_pc: Maximum z-coordinate
            
        Raises:
        -------
        ValueError
            If no clusters have been generated yet
        """
        if self.clusters is None:
            raise ValueError("No clusters generated. Call generate_clusters() first.")
        
        stats = {
            'n_clusters': len(self.clusters),
            'total_mass_msun': self.clusters['mass_msun'].sum(),
            'mean_mass_msun': self.clusters['mass_msun'].mean(),
            'median_mass_msun': self.clusters['mass_msun'].median(),
            'std_mass_msun': self.clusters['mass_msun'].std(),
            'min_mass_msun': self.clusters['mass_msun'].min(),
            'max_mass_msun': self.clusters['mass_msun'].max(),
            'mean_radius_pc': self.clusters['r_pc'].mean(),
            'max_radius_pc': self.clusters['r_pc'].max(),
            'mass_range_decades': np.log10(
                self.clusters['mass_msun'].max() / 
                self.clusters['mass_msun'].min()
            )
        }
        
        # Add velocity statistics if available
        if 'v_mag' in self.clusters.columns:
            stats['mean_v_mag'] = self.clusters['v_mag'].mean()
            stats['std_v_mag'] = self.clusters['v_mag'].std()
        
        # Add z-coordinate and vertical velocity statistics if available
        if 'z_pc' in self.clusters.columns:
            stats['min_z_pc'] = self.clusters['z_pc'].min()
            stats['max_z_pc'] = self.clusters['z_pc'].max()
            stats['mean_z_pc'] = self.clusters['z_pc'].mean()
        
        if 'vz' in self.clusters.columns:
            stats['mean_vz'] = self.clusters['vz'].mean()
            stats['std_vz'] = self.clusters['vz'].std()
        
        return stats
    
    def set_velocity_profile(self, velocity_func: callable) -> None:
        """
        Set new velocity magnitudes based on a function of radius.
        
        This recalculates vx and vy to maintain counterclockwise motion
        with the new velocity magnitude. The vertical velocity (vz) is not affected.
        
        Parameters:
        -----------
        velocity_func : callable
            Function that takes radius (in parsecs) and returns velocity magnitude.
            Should have signature: velocity_func(r) -> velocity
            Can be a lambda function or any callable.
            
        Raises:
        -------
        ValueError
            If no clusters have been generated yet
            
        Example:
        --------
        >>> generator.generate_clusters(seed=42)
        >>> 
        >>> # Constant velocity
        >>> generator.set_velocity_profile(lambda r: 220.0)
        >>> 
        >>> # Flat rotation curve with inner rise
        >>> def rotation_curve(r):
        ...     r_scale = 200  # parsecs
        ...     v_max = 220    # km/s or velocity units
        ...     return v_max * np.tanh(r / r_scale)
        >>> generator.set_velocity_profile(rotation_curve)
        """
        if self.clusters is None:
            raise ValueError("No clusters generated. Call generate_clusters() first.")
        
        # Get positions
        x = self.clusters['x_pc'].values
        y = self.clusters['y_pc'].values
        r = self.clusters['r_pc'].values
        
        # Calculate new velocity magnitudes based on radius
        v_mag_new = velocity_func(r)
        
        # Ensure v_mag_new is an array
        if np.isscalar(v_mag_new):
            v_mag_new = np.full_like(r, v_mag_new)
        
        # Recalculate vx and vy for counterclockwise motion with new magnitude
        vx_new = np.zeros_like(x)
        vy_new = np.zeros_like(y)
        
        non_zero = r > 0
        vx_new[non_zero] = -y[non_zero] / r[non_zero] * v_mag_new[non_zero]
        vy_new[non_zero] = x[non_zero] / r[non_zero] * v_mag_new[non_zero]
        
        # Update the DataFrame
        self.clusters['vx'] = vx_new
        self.clusters['vy'] = vy_new
        self.clusters['v_mag'] = v_mag_new
        
        print(f"Velocities updated with new profile")
        print(f"Mean velocity magnitude: {v_mag_new.mean():.3f}")
        print(f"Velocity range: [{v_mag_new.min():.3f}, {v_mag_new.max():.3f}]")
        
    def plot_distributions(self, figsize: Tuple[int, int] = (16, 6),
                        save_fig: bool = False, 
                        filename: Optional[str] = None) -> None:
        """
        Create visualization plots of the cluster distributions.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size (width, height)
        save_fig : bool
            Whether to save the figure
        filename : Optional[str]
            Output filename if saving
            
        Raises:
        -------
        ValueError
            If no clusters have been generated yet
        """
        if self.clusters is None:
            raise ValueError("No clusters generated. Call generate_clusters() first.")
        
        stats = self.get_statistics()
        self.visualizer.plot_distributions(
            self.clusters, self.params, stats, 
            figsize, save_fig, filename
        )
    
    def save_to_txt(self, filename: Optional[str] = None) -> None:
        """
        Save cluster data to text file in the specified format.
        
        Parameters:
        -----------
        filename : Optional[str]
            Output filename. If None, uses default naming.
            
        Raises:
        -------
        ValueError
            If no clusters have been generated yet
            
        Format:
        -------
        # x,  y,  z,   vx,  vy,  vz, t_min, cl_mass(M_sol)
        1.0 0.0 0.0  0.0  9.496767343529157  0.0   0.0    1e4
        """
        if self.clusters is None:
            raise ValueError("No clusters generated. Call generate_clusters() first.")
        
        if filename is None:
            filename = f'star_clusters_{self.params.n_clusters}.txt'
        
        with open(filename, 'w') as f:
            # Write header
            f.write("# x,  y,  z,   vx,  vy,  vz, t_min, cl_mass(M_sol)\n")
            
            # Write data for each cluster
            for _, row in self.clusters.iterrows():
                # Format each value appropriately
                x = row['x_pc'] / 1000.0  # Convert to kpc
                y = row['y_pc'] / 1000.0  # Convert to kpc 
                z = row['z_pc'] / 1000.0  # Convert to kpc
                vx = row['vx']
                vy = row['vy']
                vz = row['vz']
                t_min = row['t_min']
                mass = row['mass_msun']
                
                # Format the line with proper spacing
                # Using format specifiers to control alignment and precision
                line = f"{x:15.8f} {y:15.8f} {z:15.8f}  " \
                    f"{vx:15.8f}  {vy:15.8f} {vz:15.8f}   " \
                    f"{t_min:15.8f}    {mass:.8e}\n"
                
                f.write(line)
        
        print(f"Cluster data saved to {filename}")
    
    def __repr__(self) -> str:
        """String representation of the generator."""
        status = "ready" if self.clusters is None else f"{len(self.clusters)} clusters generated"
        return (f"StarClusterGenerator(n={self.params.n_clusters}, "
                f"α={self.params.alpha}, radius={self.params.radius_kpc} kpc, "
                f"status={status})")