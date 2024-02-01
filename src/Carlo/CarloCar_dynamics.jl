using Parameters

@with_kw mutable struct CarloCar
    center = [60, 60] # this is x, y
    heading = 0.0
    movable = true
    size = [4.0, 2.0]
    color = "ghost white"
    collidable = true
    friction = 0.06
    velocity = [0,0] # this is xp, yp
    acceleration = 0 # this is vp (or speedp)
    angular_velocity = 0 # this is headingp
    inputSteering = 0
    inputAcceleration = 0
    max_speed = Inf
    min_speed = 0
end

function carlo_tick!(state::CarloCar; dt::Number)
    speed = sqrt(sum(state.velocity.^2))
    heading = state.heading

    # Kinematic bicycle model dynamics based on
    # "Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design" by
    # Jason Kong, Mark Pfeiffer, Georg Schildbach, Francesco Borrelli
    lr = maximum(state.size) / 2.
    lf = lr # we assume the center of mass is the same as the geometric center of the entity
    beta = atan(lr / (lf + lr) * tan(state.inputSteering))

    new_angular_velocity = speed * state.inputSteering # this is not needed and used for this model, but let's keep it for consistency (and to avoid if-else statements)
    new_acceleration = state.inputAcceleration - state.friction
    new_speed = clamp(speed + new_acceleration * dt, state.min_speed, state.max_speed)
    new_heading = heading + ((speed + new_speed)/lr)*sin(beta)*dt/2.
    angle = (heading + new_heading)/2. + beta
    new_center = state.center + (speed + new_speed)*[cos(angle), sin(angle)]*dt / 2.
    new_velocity = [new_speed * cos(new_heading), new_speed * sin(new_heading)]

    """
    # Point-mass dynamics based on
    # "Active Preference-Based Learning of Reward Functions" by
    # Dorsa Sadigh, Anca D. Dragan, S. Shankar Sastry, Sanjit A. Seshia

    new_angular_velocity = speed * state.inputSteering
    new_acceleration = state.inputAcceleration - state.friction * speed

    new_heading = heading + (state.angular_velocity + new_angular_velocity) * dt / 2.
    new_speed = clamp(speed + (state.acceleration + new_acceleration) * dt / 2., state.min_speed, state.max_speed)

    new_velocity = CartesianIndex(((speed + new_speed) / 2.) * cos((new_heading + heading) / 2.),
                            ((speed + new_speed) / 2.) * sin((new_heading + heading) / 2.))

    new_center = state.center + (state.velocity + new_velocity) * dt / 2.

    """

    state.center = new_center
    state.heading = mod(new_heading, 2*pi) # wrap the heading angle between 0 and +2pi
    state.velocity = new_velocity
    state.acceleration = new_acceleration
    state.angular_velocity = new_angular_velocity

    return state
end
