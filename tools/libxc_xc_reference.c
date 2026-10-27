// Generate the independent unpolarized PBE point-reference fixture.
//
// Build on Homebrew macOS with:
// clang tools/libxc_xc_reference.c -I/opt/homebrew/include \
//   -L/opt/homebrew/lib -lxc -o /tmp/libxc_xc_reference

#include <stdio.h>
#include <xc.h>
#include <xc_funcs.h>

#define POINT_COUNT 7

int main(void) {
    const double rho[POINT_COUNT] = {
        1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0, 1.0e-2, 1.0e-1,
    };
    const double sigma[POINT_COUNT] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-4, 1.0e-2,
    };
    xc_func_type exchange;
    xc_func_type correlation;
    if (xc_func_init(&exchange, XC_GGA_X_PBE, XC_UNPOLARIZED) != 0 ||
        xc_func_init(&correlation, XC_GGA_C_PBE, XC_UNPOLARIZED) != 0) {
        return 1;
    }

    double ex[POINT_COUNT], vxrho[POINT_COUNT], vxsigma[POINT_COUNT];
    double ec[POINT_COUNT], vcrho[POINT_COUNT], vcsigma[POINT_COUNT];
    xc_gga_exc_vxc(&exchange, POINT_COUNT, rho, sigma, ex, vxrho, vxsigma);
    xc_gga_exc_vxc(&correlation, POINT_COUNT, rho, sigma, ec, vcrho, vcsigma);

    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"generator\": \"Libxc %s\",\n", xc_version_string());
    printf("  \"exchange_functional\": \"GGA_X_PBE (101)\",\n");
    printf("  \"correlation_functional\": \"GGA_C_PBE (130)\",\n");
    printf("  \"spin\": \"unpolarized\",\n");
    printf("  \"points\": [\n");
    for (size_t i = 0; i < POINT_COUNT; ++i) {
        printf("    {\"density_bohr3\": %.17g, \"sigma_bohr8\": %.17g, "
               "\"epsilon_xc_ha\": %.17g, \"vrho_ha\": %.17g, "
               "\"vsigma_ha_bohr5\": %.17g}%s\n",
               rho[i], sigma[i], ex[i] + ec[i], vxrho[i] + vcrho[i],
               vxsigma[i] + vcsigma[i], i + 1 == POINT_COUNT ? "" : ",");
    }
    printf("  ]\n}\n");

    xc_func_end(&exchange);
    xc_func_end(&correlation);
    return 0;
}
