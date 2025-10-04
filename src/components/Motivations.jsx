import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import { ScrollTrigger } from "gsap/all";

import AnimatedTitle from "./AnimatedTitle";

gsap.registerPlugin(ScrollTrigger);

const Motivations = () => {
  useGSAP(() => {
    const clipAnimation = gsap.timeline({
      scrollTrigger: {
        trigger: "#clip",
        start: "center center",
        end: "+=800 center",
        scrub: 0.5,
        pin: true,
        pinSpacing: true,
      },
    });

    clipAnimation.to(".mask-clip-path", {
      width: "100vw",
      height: "100vh",
      borderRadius: 0,
    });
  });

  return (
    <div className="min-h-screen w-full overflow-hidden">
      <div className="relative mb-8 mt-36 flex flex-col items-center gap-5">
        <AnimatedTitle
          title="Datacenters are heating up"
          containerClass="mt-5 !text-black text-center"
        />

        <AnimatedTitle
          title="and we're paying the price"
          containerClass="mt-5 !text-black text-center"
        />
        <div className="about-subtext">
          <p className="text-xl font-bold text-black mb-4 -mt-4">
            With computers producing massive heat and throttling GPUs, big corps
            are spending 40% of their energy just for cooling
          </p>
          <p className="text-gray-500 text-xl">
            Google's datacenters alone use enough energy to power all of Ireland
          </p>
        </div>
      </div>

      <div className="h-dvh w-full" id="clip">
        <div className="mask-clip-path about-image">
          <video
            src="videos/vid2.mp4"
            autoPlay
            loop
            muted
            className="absolute left-0 top-0 size-full object-cover"
          />
        </div>
      </div>
    </div>
  );
};

export default Motivations;
