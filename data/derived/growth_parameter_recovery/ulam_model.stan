data{
    int n_species;
    int n_dish;
    int n_obs;
    vector[960] R_obs;
    vector[960] time_days;
    vector[60] is_15C_dish;
    int dish_obs[960];
    int species_dish[60];
    real max_radius_mm;
}
parameters{
    vector[n_species] alpha;
    vector[n_species] gamma;
    vector[n_dish] z_k;
    vector[n_dish] z_tau;
    real mu_log_k;
    real<lower=0> sigma_species_log_k;
    real<lower=0> sigma_dish_log_k;
    real beta_T_15;
    real mu_log_tau;
    real<lower=0> sigma_species_log_tau;
    real<lower=0> sigma_dish_log_tau;
    real delta_T_15;
    real bias_mm;
    real<lower=0> sigma_seg_mm;
}
model{
    vector[n_obs] mu;
    sigma_seg_mm ~ normal( 0 , 2 );
    bias_mm ~ normal( 0 , 0.5 );
    delta_T_15 ~ normal( 0 , 0.35 );
    sigma_dish_log_tau ~ normal( 0 , 0.2 );
    sigma_species_log_tau ~ normal( 0 , 0.35 );
    mu_log_tau ~ normal( log(3) , 0.45 );
    beta_T_15 ~ normal( 0 , 0.35 );
    sigma_dish_log_k ~ normal( 0 , 0.2 );
    sigma_species_log_k ~ normal( 0 , 0.4 );
    mu_log_k ~ normal( log(3) , 0.55 );
    z_tau ~ normal( 0 , 1 );
    z_k ~ normal( 0 , 1 );
    gamma ~ normal( mu_log_tau , sigma_species_log_tau );
    alpha ~ normal( mu_log_k , sigma_species_log_k );
    for ( i in 1:n_obs ) {
        mu[i] = fmin(max_radius_mm, elt_multiply(exp(alpha[species_dish[dish_obs[i]]] + beta_T_15 * is_15C_dish[dish_obs[i]] + z_k[dish_obs[i]] * sigma_dish_log_k), fmax(0, time_days[i] - exp(gamma[species_dish[dish_obs[i]]] + delta_T_15 * is_15C_dish[dish_obs[i]] + z_tau[dish_obs[i]] * sigma_dish_log_tau)))) + bias_mm;
    }
    R_obs ~ normal( mu , sigma_seg_mm );
}


