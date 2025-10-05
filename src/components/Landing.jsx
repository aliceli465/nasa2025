import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import { ScrollTrigger } from "gsap/all";
import { TiLocationArrow } from "react-icons/ti";

import Button from "./Button";

gsap.registerPlugin(ScrollTrigger);

const Landing = () => {
  useGSAP(() => {
    gsap.set("#video-frame", {
      clipPath: "polygon(14% 0, 72% 0, 88% 90%, 0 95%)",
      borderRadius: "0% 0% 40% 10%",
    });
    gsap.from("#video-frame", {
      clipPath: "polygon(0% 0%, 100% 0%, 100% 100%, 0% 100%)",
      borderRadius: "0% 0% 0% 0%",
      ease: "power1.inOut",
      scrollTrigger: {
        trigger: "#video-frame",
        start: "center center",
        end: "bottom center",
        scrub: true,
      },
    });
  });

  return (
    <div className="relative h-dvh w-full overflow-x-hidden">
      <div
        id="video-frame"
        className="relative z-10 h-dvh w-full overflow-hidden rounded-lg bg-blue-75"
      >
        <video
          src="videos/vid1.mp4"
          autoPlay
          loop
          muted
          className="absolute left-0 top-0 size-full object-cover object-center"
        />

        <div className="absolute left-0 top-0 z-40 size-full">
          <div className="mt-24 px-5 sm:px-10">
            <h1 className="special-font hero-heading text-[#dfdff2]">
              Le<b>Da</b>ta
            </h1>
            <br />
            <p className="mb-5 font-robert-regular text-[#dfdff2]">
              AI scheduling, edge computing, and frictionless scaling
              <br />
              Thats the LEO advantage
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Landing;
