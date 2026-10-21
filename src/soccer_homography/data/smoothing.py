from dataclasses import dataclass
from math import hypot
from typing import Protocol

import numpy as np
from filterpy.kalman import KalmanFilter


CHI_SQUARED_2D_99_PERCENT = 9.21
MAX_CONSECUTIVE_REJECTIONS = 3


@dataclass( frozen=True, slots=True )
class PositionEstimate:
  position: tuple[ float, float ]
  velocity: tuple[ float, float ]


class PositionSmoother( Protocol ):

  def reset( self ) -> None:
    ...

  def update( self, position: tuple[ float, float ], dt: float ) -> PositionEstimate:
    ...


class ExponentialSmoother:

  def __init__( self, alpha: float = 0.2, dead_zone: float = 0.15 ):
    if not 0 <= alpha <= 1:
      raise ValueError( "Alpha must be between zero and one" )
    if dead_zone < 0:
      raise ValueError( "Dead zone cannot be negative" )
    self.alpha = alpha
    self.dead_zone = dead_zone
    self._position: tuple[ float, float ] | None = None

  def reset( self ) -> None:
    self._position = None

  def update( self, position: tuple[ float, float ], dt: float ) -> PositionEstimate:
    if dt <= 0:
      raise ValueError( "Elapsed time must be positive" )

    if self._position is None:
      self._position = position
      return PositionEstimate( position, ( 0.0, 0.0 ) )

    previous = self._position
    delta_x = position[ 0 ] - previous[ 0 ]
    delta_y = position[ 1 ] - previous[ 1 ]
    if hypot( delta_x, delta_y ) < self.dead_zone:
      return PositionEstimate( previous, ( 0.0, 0.0 ) )

    smoothed = (
        self.alpha * position[ 0 ] + ( 1 - self.alpha ) * previous[ 0 ],
        self.alpha * position[ 1 ] + ( 1 - self.alpha ) * previous[ 1 ],
    )
    self._position = smoothed
    velocity = ( ( smoothed[ 0 ] - previous[ 0 ] ) / dt, ( smoothed[ 1 ] - previous[ 1 ] ) / dt )
    return PositionEstimate( smoothed, velocity )


class KalmanSmoother:

  def __init__( self, max_speed: float = 10.0, measurement_noise: float = 0.25, acceleration_variance: float = 1.0 ):
    if max_speed <= 0 or measurement_noise <= 0 or acceleration_variance <= 0:
      raise ValueError( "Kalman smoother parameters must be positive" )
    self.max_speed = max_speed
    self.measurement_noise = measurement_noise
    self.acceleration_variance = acceleration_variance
    self._filter: KalmanFilter | None = None
    self._consecutive_rejections = 0

  def reset( self ) -> None:
    self._filter = None
    self._consecutive_rejections = 0

  def update( self, position: tuple[ float, float ], dt: float ) -> PositionEstimate:
    if dt <= 0:
      raise ValueError( "Elapsed time must be positive" )

    measurement = np.asarray( position, dtype=float ).reshape( 2, 1 )
    if self._filter is None:
      kf = KalmanFilter( dim_x=4, dim_z=2 )
      kf.x = np.array( [ position[ 0 ], position[ 1 ], 0.0, 0.0 ] ).reshape( 4, 1 )
      kf.P = np.diag( [ 1.0, 1.0, 25.0, 25.0 ] )
      kf.H = np.array( [ [ 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0 ] ] )
      kf.R = np.eye( 2 ) * self.measurement_noise
      self._filter = kf
      self._consecutive_rejections = 0
      return PositionEstimate( position, ( 0.0, 0.0 ) )

    kf = self._filter
    kf.F = np.array( [
        [ 1.0, 0.0, dt, 0.0 ],
        [ 0.0, 1.0, 0.0, dt ],
        [ 0.0, 0.0, 1.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 1.0 ],
    ] )
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    variance = self.acceleration_variance
    kf.Q = variance * np.array( [
        [ dt4 / 4, 0.0, dt3 / 2, 0.0 ],
        [ 0.0, dt4 / 4, 0.0, dt3 / 2 ],
        [ dt3 / 2, 0.0, dt2, 0.0 ],
        [ 0.0, dt3 / 2, 0.0, dt2 ],
    ] )
    kf.predict()

    innovation = measurement - kf.x[ :2 ]
    innovation_covariance = kf.H @ kf.P @ kf.H.T + kf.R
    mahalanobis_distance_squared = float(
        ( innovation.T @ np.linalg.solve( innovation_covariance, innovation ) )[ 0, 0 ]
    )
    if mahalanobis_distance_squared <= CHI_SQUARED_2D_99_PERCENT:
      kf.update( measurement )
      self._consecutive_rejections = 0
    else:
      self._consecutive_rejections += 1
      if self._consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
        kf.x = np.array( [ position[ 0 ], position[ 1 ], 0.0, 0.0 ] ).reshape( 4, 1 )
        kf.P = np.diag( [ 1.0, 1.0, 25.0, 25.0 ] )
        self._consecutive_rejections = 0
        return PositionEstimate( position, ( 0.0, 0.0 ) )

    velocity = kf.x[ 2:4, 0 ]
    speed = np.linalg.norm( velocity )
    if speed > self.max_speed:
      kf.x[ 2:4 ] *= self.max_speed / speed

    return PositionEstimate(
        ( float( kf.x[ 0, 0 ] ), float( kf.x[ 1, 0 ] ) ),
        ( float( kf.x[ 2, 0 ] ), float( kf.x[ 3, 0 ] ) ),
    )
