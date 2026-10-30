// Generate the independent spin-polarized LDA-PZ point-reference fixture.
//
// Build on Homebrew macOS with:
// clang tools/libxc_lsda_reference.c -I/opt/homebrew/include \
//   -L/opt/homebrew/lib -lxc -o /tmp/libxc_lsda_reference

#include <stdio.h>
#include <xc.h>
#include <xc_funcs.h>

#define POINT_COUNT 7

int main(void) {
    // Libxc stores polarized densities as [up_0, down_0, up_1, down_1, ...].
    const double rho[2 * POINT_COUNT] = {
        5.0e-7, 5.0e-7,
        8.0e-5, 2.0e-5,
        6.0e-3, 4.0e-3,
        9.0e-2, 1.0e-2,
        7.0e-1, 3.0e-1,
        1.0e-1, 0.0,
        0.0, 1.0e-1,
    };
    xc_func_type exchange;
    xc_func_type correlation;
    if (xc_func_init(&exchange, XC_LDA_X, XC_POLARIZED) != 0 ||
        xc_func_init(&correlation, XC_LDA_C_PZ, XC_POLARIZED) != 0) {
        return 1;
    }

    double ex[POINT_COUNT], vx[2 * POINT_COUNT];
    double ec[POINT_COUNT], vc[2 * POINT_COUNT];
    xc_lda_exc_vxc(&exchange, POINT_COUNT, rho, ex, vx);
    xc_lda_exc_vxc(&correlation, POINT_COUNT, rho, ec, vc);

    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"generator\": \"Libxc %s\",\n", xc_version_string());
    printf("  \"exchange_functional\": \"LDA_X (1)\",\n");
    printf("  \"correlation_functional\": \"LDA_C_PZ (9)\",\n");
    printf("  \"spin\": \"polarized\",\n");
    printf("  \"points\": [\n");
    for (size_t i = 0; i < POINT_COUNT; ++i) {
        printf("    {\"density_up_bohr3\": %.17g, \"density_down_bohr3\": %.17g, "
               "\"epsilon_xc_ha\": %.17g, \"v_up_ha\": %.17g, "
               "\"v_down_ha\": %.17g}%s\n",
               rho[2 * i], rho[2 * i + 1], ex[i] + ec[i],
               vx[2 * i] + vc[2 * i], vx[2 * i + 1] + vc[2 * i + 1],
               i + 1 == POINT_COUNT ? "" : ",");
    }
    printf("  ]\n}\n");

    xc_func_end(&exchange);
    xc_func_end(&correlation);
    return 0;
}
